# -*- coding: utf-8 -*-
import logging
import queue
import multiprocessing as mp
from multiprocessing import Queue
import math
import gc
from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict, Literal, Optional, cast

import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available
import nltk

# Download the 'punkt' tokenizer models if not already present
nltk.download('punkt', quiet=True)

try:
    import torch_musa
except ImportError:
    pass

logger = logging.getLogger(__name__)


class AbsEmbedder(ABC):
    """
    Base class for embedder.
    Extend this class and implement :meth:`encode_queries`, :meth:`encode_corpus`, :meth:`encode` for custom embedders.

    Args:
        model_name_or_path (str): If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
            load a model from HuggingFace Hub with the name.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to :data:`True`.
        use_fp16 (bool, optional): If true, use half-precision floating-point to speed up computation with a slight performance
            degradation. Defaults to :data:`True`.
        query_instruction_for_retrieval: (Optional[str], optional): Query instruction for retrieval tasks, which will be used with
            with :attr:`query_instruction_format`. Defaults to :data:`None`.
        query_instruction_format: (str, optional): The template for :attr:`query_instruction_for_retrieval`. Defaults to :data:`"{}{}"`.
        devices (Optional[Union[str, int, List[str], List[int]]], optional): Devices to use for model inference. Defaults to :data:`None`.
        batch_size (int, optional): Batch size for inference. Defaults to :data:`256`.
        query_max_length (int, optional): Maximum length for query. Defaults to :data:`512`.
        passage_max_length (int, optional): Maximum length for passage. Defaults to :data:`512`.
        convert_to_numpy (bool, optional): If True, the output embedding will be a Numpy array. Otherwise, it will be a Torch Tensor.
            Defaults to :data:`True`.
        kwargs (Dict[Any], optional): Additional parameters for HuggingFace Transformers config or children classes.
    """

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}",
        devices: Optional[Union[str, int, List[str], List[int]]] = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ):
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.target_devices = self.get_target_devices(devices)

        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.convert_to_numpy = convert_to_numpy

        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.kwargs = kwargs

        self.tokenizer = None
        self.model = None
        self.pool = None

    def stop_self_pool(self):
        if self.pool is not None:
            self.stop_multi_process_pool(self.pool)
            self.pool = None
        try:
            self.model.to('cpu')
            torch.cuda.empty_cache()
        except Exception:
            pass
        if gc is not None and callable(gc.collect):
            gc.collect()

    @staticmethod
    def get_target_devices(devices: Union[str, int, List[str], List[int]]) -> List[str]:
        """
        Args:
            devices (Union[str, int, List[str], List[int]]): specified devices, can be `str`, `int`, list of `str`, or list of `int`.

        Raises:
            ValueError: Devices should be a string or an integer or a list of strings or a list of integers.

        Returns:
            List[str]: A list of target devices in format.
        """
        if devices is None:
            if torch.cuda.is_available():
                return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                return [f"npu:{i}" for i in range(torch.npu.device_count())]
            elif hasattr(torch, "musa") and torch.musa.is_available():
                return [f"musa:{i}" for i in range(torch.musa.device_count())]
            elif torch.backends.mps.is_available():
                try:
                    return [f"mps:{i}" for i in range(torch.mps.device_count())]
                except Exception:
                    return ["mps"]
            else:
                return ["cpu"]
        elif isinstance(devices, str):
            return [devices]
        elif isinstance(devices, int):
            if hasattr(torch, "musa") and torch.musa.is_available():
                return [f"musa:{devices}"]
            else:
                return [f"cuda:{devices}"]
        elif isinstance(devices, list):
            if isinstance(devices[0], str):
                return devices
            elif isinstance(devices[0], int):
                if hasattr(torch, "musa") and torch.musa.is_available():
                    return [f"musa:{device}" for device in devices]
                else:
                    return [f"cuda:{device}" for device in devices]
            else:
                raise ValueError("devices should be a string or an integer or a list of strings or a list of integers.")
        else:
            raise ValueError("devices should be a string or an integer or a list of strings or a list of integers.")

    @staticmethod
    def get_detailed_instruct(instruction_format: str, instruction: str, sentence: str):
        """Combine the instruction and sentence along with the instruction format.

        Args:
            instruction_format (str): Format for instruction.
            instruction (str): The text of instruction.
            sentence (str): The sentence to concatenate with.

        Returns:
            str: The complete sentence with instruction
        """
        if "\\n" in instruction_format:
            instruction_format = instruction_format.replace("\\n", "\n")
        return instruction_format.format(instruction, sentence)

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):
        """encode the queries using the instruction if provided.

        Args:
            queries (Union[List[str], str]): Input queries to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will
                be a Torch Tensor. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: Return the embedding vectors in a numpy array or tensor.
        """
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.query_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=self.query_instruction_for_retrieval,
            instruction_format=self.query_instruction_format,
            **kwargs
        )

    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):
        """encode the corpus using the instruction if provided.

        Args:
            corpus (Union[List[str], str]): Input corpus to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will
                be a Torch Tensor. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: Return the embedding vectors in a numpy array or tensor.
        """
        passage_instruction_for_retrieval = self.kwargs.get("passage_instruction_for_retrieval", None)
        passage_instruction_format = self.kwargs.get("passage_instruction_format", "{}{}")

        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.passage_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            corpus,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=passage_instruction_for_retrieval,
            instruction_format=passage_instruction_format,
            **kwargs
        )

    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        instruction: Optional[str] = None,
        instruction_format: Optional[str] = None,
        **kwargs: Any
    ):
        """encode the input sentences with the embedding model.

        Args:
            sentences (Union[List[str], str]): Input sentences to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will
                be a Torch Tensor. Defaults to :data:`None`.
            instruction (Optional[str], optional): The text of instruction. Defaults to :data:`None`.
            instruction_format (Optional[str], optional): Format for instruction. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.passage_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        if instruction is not None:
            if isinstance(sentences, str):
                sentences = self.get_detailed_instruct(instruction_format, instruction, sentences)
            else:
                sentences = [self.get_detailed_instruct(instruction_format, instruction, sentence) for sentence in
                             sentences]

        if isinstance(sentences, str) or len(self.target_devices) == 1:
            return self.encode_single_device(
                sentences,
                batch_size=batch_size,
                max_length=max_length,
                convert_to_numpy=convert_to_numpy,
                device=self.target_devices[0],
                **kwargs
            )

        if self.pool is None:
            self.pool = self.start_multi_process_pool(AbsEmbedder._encode_multi_process_worker)
        embeddings = self.encode_multi_process(
            sentences,
            self.pool,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
        return embeddings

    def __del__(self):
        self.stop_self_pool()

    @abstractmethod
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        This method should encode sentences and return embeddings on a single device.
        """
        pass

    def start_multi_process_pool(
        self,
        process_target_func: Any,
    ) -> Dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes.
        This method is recommended for encoding on multiple GPUs or CPUs.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, self.target_devices))))

        self.model.to("cpu")
        self.model.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in tqdm(self.target_devices, desc='Initializing target devices'):
            p = ctx.Process(
                target=process_target_func,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    @staticmethod
    def _encode_multi_process_worker(
        target_device: str, model: 'AbsEmbedder', input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in a multi-process setup.
        """
        while True:
            try:
                chunk_id, sentences, kwargs = (
                    input_queue.get()
                )
                embeddings = model.encode_single_device(
                    sentences,
                    device=target_device,
                    **kwargs
                )

                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    @staticmethod
    def stop_multi_process_pool(pool: Dict[Literal["input", "output", "processes"], Any]) -> None:
        """
        Stops all processes started with start_multi_process_pool.
        """
        for p in pool["processes"]:
            p.terminate()

        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()

    def encode_multi_process(
        self,
        sentences: List[str],
        pool: Dict[Literal["input", "output", "processes"], Any],
        **kwargs
    ):
        chunk_size = math.ceil(len(sentences) / len(pool["processes"]))

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [last_chunk_id, chunk, kwargs]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Processing chunks")],
            key=lambda x: x[0],
        )
        embeddings = self._concatenate_results_from_multi_process([result[1] for result in results_list])
        return embeddings

    def _concatenate_results_from_multi_process(self, results_list: List[Union[torch.Tensor, np.ndarray, Any]]):
        """Concatenate and return the results from all the processes."""
        if isinstance(results_list[0], torch.Tensor):
            results_list = [res.to(self.target_devices[0]) for res in results_list]
            return torch.cat(results_list, dim=0)
        elif isinstance(results_list[0], np.ndarray):
            return np.concatenate(results_list, axis=0)
        else:
            raise NotImplementedError(f"Unsupported type for results_list: {type(results_list[0])}")


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Last token pooling method.

    Args:
        last_hidden_states (torch.Tensor): The last hidden state of the model.
        attention_mask (torch.Tensor): Attention mask.

    Returns:
        torch.Tensor: The embedding vectors after pooling.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class LMKEmbedder(AbsEmbedder):
    """
    LMKEmbedder for sentence-level retrieval from passages.
    This embedder splits passages into sentences and appends an EOS token to each sentence/query
    to get its representation from the last token's hidden state.
    """
    DEFAULT_POOLING_METHOD = "last_token"

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "Instruct: {}\nQuery: {}",
        devices: Optional[Union[str, List[str]]] = None,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )

        # For decoder-only models, the padding token is often not set.
        # We use the EOS token as the padding token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if self.kwargs.get("pooling_method", "last_token") != "last_token":
            raise ValueError("Pooling method must be 'last_token' for LMKEmbedder.")

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode queries by appending the EOS token to each query.
        """
        if isinstance(queries, str):
            processed_queries = [queries + self.tokenizer.eos_token]
        else:
            processed_queries = [q + self.tokenizer.eos_token for q in queries]

        return super().encode_queries(
            processed_queries,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode corpus by splitting passages into sentences and appending the EOS token to each sentence.
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        all_sentences = []
        for passage in corpus:
            sents = nltk.sent_tokenize(passage)
            all_sentences.extend(sents)

        if not all_sentences:
            emb_dim = self.model.config.hidden_size
            if convert_to_numpy is None:
                convert_to_numpy = self.convert_to_numpy

            if convert_to_numpy:
                return np.empty((0, emb_dim), dtype=np.float32)
            else:
                dtype = torch.float16 if self.use_fp16 else torch.float32
                return torch.empty((0, emb_dim), dtype=dtype)

        processed_sentences = [s + self.tokenizer.eos_token for s in all_sentences]

        return super().encode_corpus(
            processed_sentences,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any
    ):
        """Encode input sentences on a single device."""
        if device is None:
            device = self.target_devices[0]

        if "cpu" in device: self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_inputs = []
        for start_index in trange(0, len(sentences), batch_size, desc='Tokenizing',
                                  disable=len(sentences) < batch_size):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                **kwargs
            )
            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            all_inputs.extend(inputs_batch)

        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # Dynamically reduce batch size in case of OOM
        try_batch_size = batch_size
        while try_batch_size > 0:
            try:
                inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[:try_batch_size],
                    padding=True,
                    return_tensors='pt',
                    **kwargs
                ).to(device)
                self.model(**inputs_batch, return_dict=True).last_hidden_state
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    try_batch_size //= 2
                    logger.warning(f"CUDA OOM. Reducing batch size to {try_batch_size}")
                else:
                    raise e
        if try_batch_size == 0:
            raise RuntimeError("Batch size is 0. Cannot proceed.")
        batch_size = try_batch_size


        all_embeddings = []
        for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size), desc="Embedding",
                                disable=len(sentences) < batch_size):
            inputs_batch_data = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch_data,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)

            last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
            embeddings = last_token_pool(last_hidden_state, inputs_batch['attention_mask'])

            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Restore original order
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]

        if convert_to_numpy:
            all_embeddings = all_embeddings.numpy()

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


if __name__ == "__main__":
    try:
        import nltk
        nltk.download('punkt_tab')
        # NOTE: Replace the model path
        model = LMKEmbedder(
            "your-model-name-or-path",
            use_fp16=True,
            devices=['cuda:0']
        )

        queries = ['What is the primary color of a ripe banana?']
        passages = [
            "The banana is a widely consumed tropical fruit that belongs to the genus Musa. It originates from "
            "Southeast Asia and has been cultivated for thousands of years. The fruit grows in large hanging clusters "
            "called hands, with individual bananas known as fingers. Bananas undergo a fascinating ripening process "
            "where the primary color transforms from green to yellow as chlorophyll breaks down and carotenoids "
            "become more visible. The primary color of a ripe banana is typically a bright, vibrant yellow, though "
            "some varieties may exhibit slight variations. As bananas continue to ripen beyond their peak, brown spots "
            "begin to appear and the skin gradually darkens. Before reaching full maturity, the skin maintains a "
            "solid green hue due to high chlorophyll content. Nutritionally, bananas are an excellent source of "
            "potassium, vitamin B6, and dietary fiber, making them a healthy choice for many diets. People worldwide "
            "enjoy bananas in various forms - eaten raw, blended into smoothies, baked in desserts, or sliced into "
            "breakfast cereals. In many tropical regions, bananas are also cooked as a vegetable when still green. "
            "The global banana trade represents a significant agricultural industry, with major exporters including "
            "Ecuador, the Philippines, and Costa Rica."
        ]

        # 1. Encode the query. Output shape: (1, embedding_dim)
        q_embeddings = model.encode_queries(queries, convert_to_numpy=False)

        # 2. Encode sentences in the passage. Output shape: (N, embedding_dim), where N is the number of sentences.
        p_embeddings = model.encode_corpus(passages, convert_to_numpy=False)

        # 3. Compute dot product scores. Shape: (1, N)
        scores = q_embeddings @ p_embeddings.T

        print("Scores:")
        print(scores)
        print("\nScores shape:")
        print(scores.shape)


    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        print("Please replace 'your-model-name-or-path' with a valid model identifier to run this example.")
