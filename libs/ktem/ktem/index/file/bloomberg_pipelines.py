from __future__ import annotations

import json
import logging
import pytz
import re
import shutil
import threading
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Generator, Optional, Sequence

import tiktoken
from datetime import datetime
from decouple import config
from ktem.db.models import engine
from ktem.embeddings.manager import embedding_models_manager
from ktem.llms.manager import llms
from ktem.rerankings.manager import reranking_models_manager
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from theflow.settings import settings
from theflow.utils.modules import import_dotted_string

from kotaemon.base import BaseComponent, Document, Node, Param, RetrievedDocument
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.indices import VectorIndexing, VectorRetrieval
from kotaemon.indices.ingests.files import (
    KH_DEFAULT_FILE_EXTRACTORS,
    adobe_reader,
    azure_reader,
    docling_reader,
    unstructured,
    web_reader,
)
from kotaemon.indices.rankings import BaseReranking, LLMReranking, LLMTrulensScoring
from kotaemon.indices.splitters import BaseSplitter, TokenSplitter

from .base import BaseFileIndexIndexing, BaseFileIndexRetriever

logger = logging.getLogger(__name__)


@lru_cache
def dev_settings():
    """Retrieve the developer settings from flowsettings.py"""
    file_extractors = {}

    if hasattr(settings, "FILE_INDEX_PIPELINE_FILE_EXTRACTORS"):
        file_extractors = {
            key: import_dotted_string(value, safe=False)()
            for key, value in settings.FILE_INDEX_PIPELINE_FILE_EXTRACTORS.items()
        }

    chunk_size = None
    if hasattr(settings, "FILE_INDEX_PIPELINE_SPLITTER_CHUNK_SIZE"):
        chunk_size = settings.FILE_INDEX_PIPELINE_SPLITTER_CHUNK_SIZE

    chunk_overlap = None
    if hasattr(settings, "FILE_INDEX_PIPELINE_SPLITTER_CHUNK_OVERLAP"):
        chunk_overlap = settings.FILE_INDEX_PIPELINE_SPLITTER_CHUNK_OVERLAP

    return file_extractors, chunk_size, chunk_overlap


_default_token_func = tiktoken.encoding_for_model("gpt-3.5-turbo").encode


class IndexPipeline(BaseComponent):
    """Index a single file"""

    loader: BaseReader
    splitter: BaseSplitter | None
    chunk_batch_size: int = 200

    Source = Param(help="The SQLAlchemy Source table")
    Index = Param(help="The SQLAlchemy Index table")
    VS = Param(help="The VectorStore")
    DS = Param(help="The DocStore")
    FSPath = Param(help="The file storage path")
    user_id = Param(help="The user id")
    collection_name: str = "default"
    private: bool = False
    run_embedding_in_thread: bool = False
    embedding: BaseEmbeddings

    @Node.auto(depends_on=["Source", "Index", "embedding"])
    def vector_indexing(self) -> VectorIndexing:
        return VectorIndexing(
            vector_store=self.VS, doc_store=self.DS, embedding=self.embedding
        )
    
    def extract_speakers(self, text: str) -> tuple[list[str], dict[str, str]]:
        # Define section headers
        company_headers = ['Company Participants', 'Executives', 'Corporate Participants']
        other_headers = ['Conference Call Participants', 'Analysts', 'Other Participants']
        
        # Find the start of company participants section
        company_start = None
        for header in company_headers:
            match = re.search(rf'{re.escape(header)}\n', text)
            if match:
                company_start = match.end()
                break
        
        if company_start is None:
            raise ValueError("Company participants section not found")
        
        # Find the end of company participants section (next header or end of text)
        other_headers_escaped = '|'.join([re.escape(h) for h in other_headers + ['Presentation', 'Questions And Answers']])
        next_section_pattern = rf'(?={other_headers_escaped})'
        next_section_match = re.search(next_section_pattern, text[company_start:])
        if next_section_match:
            company_end = company_start + next_section_match.start()
        else:
            company_end = len(text)
        
        company_text = text[company_start:company_end].strip()

        # Find the start of other participants section
        other_start = None
        for header in other_headers:
            match = re.search(rf'{re.escape(header)}\n', text)
            if match:
                other_start = match.end()
                break
        
        if other_start is None:
            raise ValueError("Other participants section not found")
        
        # Find the end of other participants section
        next_section_after_other = re.search(r'\n\n|Presentation|MANAGEMENT DISCUSSION SECTION', text[other_start:], re.IGNORECASE)
        if next_section_after_other:
            other_end = other_start + next_section_after_other.start()
        else:
            other_end = len(text)
        
        other_text = text[other_start:other_end].strip()

        # Extract names from both sections
        company_names = [line.split(",")[0].strip() for line in company_text.split("\n")]
        other_names = [line.split(",")[0].strip() for line in other_text.split("\n")]
        
        # Combine names and create position dictionary
        names = company_names + other_names
        position_dict = {name: 'Insider' for name in company_names}
        position_dict.update({name: 'Outsider' for name in other_names})
        
        return names, position_dict
    
    def split_content_by_speakers(self, content: str, speakers: list[str]) -> list[tuple[str, str]]:
        speaker_pattern = '|'.join(re.escape(speaker) for speaker in speakers)
        pattern = rf'(?:Q - |A - )?({speaker_pattern})(?:\s*{{BIO \d+ <GO>}})?\n((?:(?!\n(?:Q - |A - )?(?:{speaker_pattern})(?:\s*{{BIO \d+ <GO>}})?).)*)'

        result = []
        for match in re.finditer(pattern, content, re.DOTALL | re.MULTILINE):
            speaker = match.group(1).strip()
            text_chunk = match.group(2).strip()
            if not text_chunk or text_chunk == '':
                continue
            if speaker != 'Operator':
                result.append((speaker, text_chunk))
        
        return result
    
    def extract_date(self, line: str) -> Optional[str]:
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, line)
        
        if match:
            date_str = match.group(1)
            date_of_transcript = datetime.strptime(date_str, '%Y-%m-%d')

            return date_of_transcript.strftime('%B %d, %Y')
        else:
            logging.error("No date of transcript is extracted")
            return None
    
    def extract_ticker(self, file_name) -> str:
        # Define a regex pattern to match the ticker inside parentheses
        ticker_pattern = r'\((.*?)\)'
        match = re.search(ticker_pattern, file_name)
        
        if match:
            ticker = match.group(1)
            return ticker
        else:
            logging.error("No stock ticker is extracted")
            return None

    def extract_quarter(self, file_name) -> str:
        # Pattern for standard quarter format (Q1 2023, Q2 2023, etc.)
        pattern_standard = r'Q\d 2\d{3}'
        # Pattern for year format (Y 2019)
        pattern_year = r'Y 2\d{3}'
        # Pattern for semester format (S1 2019)
        pattern_semester = r'S1 2\d{3}'

        match = re.search(pattern_standard, file_name)
        if match:
            return match.group()

        match = re.search(pattern_year, file_name)
        if match:
            year = match.group()[2:]  # Extract year (e.g., 2019)
            return f'Q4 {year}'
        
        match = re.search(pattern_semester, file_name)
        if match:
            year = match.group()[3:]  # Extract year (e.g., 2019)
            return f'Q2 {year}'

        # If no pattern matches
        logging.error("No fiscal quarter is extracted")
        return None
    
    def preprocess_transcript(self, text_docs: list[Document]) -> list[Document]:
        """Assume transcript is txt, process a list of ONE Document object into chunked Document objects with updated metadata"""
        file_name = text_docs[0].metadata.get("file_name", "transcript") if text_docs else "transcript"
        
        file_content = "\n".join(doc.text for doc in text_docs)

        speakers, position_dict = self.extract_speakers(file_content)
        speakers.append('Operator')
        print(f"Speakers: {speakers}")

        chunks = self.split_content_by_speakers(file_content, speakers)
        
        date_of_transcript = self.extract_date(file_content)
        print(f"Date of transcript: {date_of_transcript}")
        
        stock_ticker = self.extract_ticker(file_content)
        
        transcript_type = 'Earnings call' if 'Earnings' in file_name else 'Others'
        fiscal_quarter = self.extract_quarter(file_content) if transcript_type == 'Earnings call' else None

        original_metadata = text_docs[0].metadata if text_docs else {}

        processed_chunks = []
        for i, (speaker, text_chunk) in enumerate(chunks):
            lines = text_chunk.split('\n')
            content = ' '.join(line.strip() for line in lines)
            chunk_number = f"Chunk {i}"
            metadata = {
                **original_metadata,
                "stock_ticker": stock_ticker,
                "speaker": speaker,
                "speaker_background": position_dict.get(speaker, ""),
                "date_of_transcript": date_of_transcript,
                "fiscal_quarter": fiscal_quarter,
                "transcript_type": transcript_type,             
                "transcript_name": f"{fiscal_quarter} - {file_name}",
                "chunk_number": chunk_number,
            }

            doc = Document(
                text=content,
                metadata=metadata
            )
            processed_chunks.append(doc)

        return processed_chunks

    def handle_docs(self, docs, file_id, file_name) -> Generator[Document, None, int]:
        s_time = time.time()
        text_docs = []
        non_text_docs = []
        thumbnail_docs = []

        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "text":
                text_docs.append(doc)
            elif doc_type == "thumbnail":
                thumbnail_docs.append(doc)
            else:
                non_text_docs.append(doc)

        print(f"Got {len(thumbnail_docs)} page thumbnails")

        # Get a list of document objects with text and metadata
        all_chunks = self.preprocess_transcript(text_docs)

        to_index_chunks = all_chunks + non_text_docs + thumbnail_docs

        # add to doc store
        chunks = []
        n_chunks = 0
        chunk_size = self.chunk_batch_size * 4
        for start_idx in range(0, len(to_index_chunks), chunk_size):
            chunks = to_index_chunks[start_idx : start_idx + chunk_size]
            self.handle_chunks_docstore(chunks, file_id)
            n_chunks += len(chunks)
            yield Document(
                f" => [{file_name}] Processed {n_chunks} chunks",
                channel="debug",
            )

        def insert_chunks_to_vectorstore():
            chunks = []
            n_chunks = 0
            chunk_size = self.chunk_batch_size
            for start_idx in range(0, len(to_index_chunks), chunk_size):
                chunks = to_index_chunks[start_idx : start_idx + chunk_size]
                self.handle_chunks_vectorstore(chunks, file_id)
                n_chunks += len(chunks)
                if self.VS:
                    yield Document(
                        f" => [{file_name}] Created embedding for {n_chunks} chunks",
                        channel="debug",
                    )

        # run vector indexing in thread if specified
        if self.run_embedding_in_thread:
            print("Running embedding in thread")
            threading.Thread(
                target=lambda: list(insert_chunks_to_vectorstore())
            ).start()
        else:
            yield from insert_chunks_to_vectorstore()

        print("indexing step took", time.time() - s_time)
        return n_chunks

    def handle_chunks_docstore(self, chunks, file_id):
        """Run chunks"""
        # run embedding, add to both vector store and doc store
        self.vector_indexing.add_to_docstore(chunks)

        # record in the index
        with Session(engine) as session:
            nodes = []
            for chunk in chunks:
                nodes.append(
                    self.Index(
                        source_id=file_id,
                        target_id=chunk.doc_id,
                        relation_type="document",
                    )
                )
            session.add_all(nodes)
            session.commit()

    def handle_chunks_vectorstore(self, chunks, file_id):
        """Run chunks"""
        # run embedding, add to both vector store and doc store
        self.vector_indexing.add_to_vectorstore(chunks)
        self.vector_indexing.write_chunk_to_file(chunks)

        if self.VS:
            # record in the index
            with Session(engine) as session:
                nodes = []
                for chunk in chunks:
                    nodes.append(
                        self.Index(
                            source_id=file_id,
                            target_id=chunk.doc_id,
                            relation_type="vector",
                        )
                    )
                session.add_all(nodes)
                session.commit()

    def get_id_if_exists(self, file_path: str | Path) -> Optional[str]:
        """Check if the file is already indexed

        Args:
            file_path: the path to the file

        Returns:
            the file id if the file is indexed, otherwise None
        """
        file_name = file_path.name if isinstance(file_path, Path) else file_path
        if self.private:
            cond: tuple = (
                self.Source.name == file_name,
                self.Source.user == self.user_id,
            )
        else:
            cond = (self.Source.name == file_name,)

        with Session(engine) as session:
            stmt = select(self.Source).where(*cond)
            item = session.execute(stmt).first()
            if item:
                return item[0].id

        return None

    def store_url(self, url: str) -> str:
        """Store URL into the database and storage, return the file id

        Args:
            url: the URL

        Returns:
            the file id
        """
        file_hash = sha256(url.encode()).hexdigest()
        source = self.Source(
            name=url,
            path=file_hash,
            size=0,
            user=self.user_id,  # type: ignore
        )
        with Session(engine) as session:
            session.add(source)
            session.commit()
            file_id = source.id

        return file_id

    def store_file(self, file_path: Path) -> str:
        """Store file into the database and storage, return the file id

        Args:
            file_path: the path to the file

        Returns:
            the file id
        """
        with file_path.open("rb") as fi:
            file_hash = sha256(fi.read()).hexdigest()

        shutil.copy(file_path, self.FSPath / file_hash)
        source = self.Source(
            name=file_path.name,
            path=file_hash,
            size=file_path.stat().st_size,
            user=self.user_id,  # type: ignore
        )
        with Session(engine) as session:
            session.add(source)
            session.commit()
            file_id = source.id

        return file_id

    def finish(self, file_id: str, file_path: str | Path) -> str:
        """Finish the indexing"""
        with Session(engine) as session:
            stmt = select(self.Source).where(self.Source.id == file_id)
            result = session.execute(stmt).first()
            if not result:
                return file_id

            item = result[0]

            # populate the number of tokens
            doc_ids_stmt = select(self.Index.target_id).where(
                self.Index.source_id == file_id,
                self.Index.relation_type == "document",
            )
            doc_ids = [_[0] for _ in session.execute(doc_ids_stmt)]
            token_func = self.get_token_func()
            if doc_ids and token_func:
                docs = self.DS.get(doc_ids)
                item.note["tokens"] = sum([len(token_func(doc.text)) for doc in docs])

            # populate the note
            item.note["loader"] = self.get_from_path("loader").__class__.__name__

            session.add(item)
            session.commit()

        return file_id

    def get_token_func(self):
        """Get the token function for calculating the number of tokens"""
        return _default_token_func

    def delete_file(self, file_id: str):
        """Delete a file from the db, including its chunks in docstore and vectorstore

        Args:
            file_id: the file id
        """
        with Session(engine) as session:
            session.execute(delete(self.Source).where(self.Source.id == file_id))
            vs_ids, ds_ids = [], []
            index = session.execute(
                select(self.Index).where(self.Index.source_id == file_id)
            ).all()
            for each in index:
                if each[0].relation_type == "vector":
                    vs_ids.append(each[0].target_id)
                elif each[0].relation_type == "document":
                    ds_ids.append(each[0].target_id)
                session.delete(each[0])
            session.commit()

        if vs_ids and self.VS:
            self.VS.delete(vs_ids)
        if ds_ids:
            self.DS.delete(ds_ids)

    def run(
        self, file_path: str | Path, reindex: bool, **kwargs
    ) -> tuple[str, list[Document]]:
        raise NotImplementedError

    def stream(
        self, file_path: str | Path, reindex: bool, **kwargs
    ) -> Generator[Document, None, tuple[str, list[Document]]]:
        # check if the file is already indexed
        if isinstance(file_path, Path):
            file_path = file_path.resolve()

        file_id = self.get_id_if_exists(file_path)

        if isinstance(file_path, Path):
            if file_id is not None:
                if not reindex:
                    raise ValueError(
                        f"File {file_path.name} already indexed. Please rerun with "
                        "reindex=True to force reindexing."
                    )
                else:
                    # remove the existing records
                    yield Document(
                        f" => Removing old {file_path.name}", channel="debug"
                    )
                    self.delete_file(file_id)
                    file_id = self.store_file(file_path)
            else:
                # add record to db
                file_id = self.store_file(file_path)
        else:
            if file_id is not None:
                raise ValueError(f"URL {file_path} already indexed.")
            else:
                # add record to db
                file_id = self.store_url(file_path)

        # extract the file
        if isinstance(file_path, Path):
            extra_info = default_file_metadata_func(str(file_path))
            file_name = file_path.name
        else:
            extra_info = {"file_name": file_path}
            file_name = file_path

        extra_info["file_id"] = file_id
        extra_info["collection_name"] = self.collection_name

        yield Document(f" => Converting {file_name} to text", channel="debug")
        docs = self.loader.load_data(file_path, extra_info=extra_info)  
        yield Document(f" => Converted {file_name} to text", channel="debug")
        yield from self.handle_docs(docs, file_id, file_name)

        self.finish(file_id, file_path)

        yield Document(f" => Finished indexing {file_name}", channel="debug")
        return file_id, docs
    
    
class IndexBloombergPipeline(BaseFileIndexIndexing):
    """Index the file. Decide which pipeline based on the file type.

    This method is essentially a factory to decide which indexing pipeline to use.

    We can decide the pipeline programmatically, and/or automatically based on an LLM.
    If we based on the LLM, essentially we will log the LLM thought process in a file,
    and then during the indexing, we will read that file to decide which pipeline
    to use, and then log the operation in that file. Overtime, the LLM can learn to
    decide which pipeline should be used.
    """

    reader_mode: str = Param("default", help="The reader mode")
    embedding: BaseEmbeddings
    run_embedding_in_thread: bool = False

    @Param.auto(depends_on="reader_mode")
    def readers(self):
        readers = deepcopy(KH_DEFAULT_FILE_EXTRACTORS)
        print("reader_mode", self.reader_mode)
        if self.reader_mode == "adobe":
            readers[".pdf"] = adobe_reader
        elif self.reader_mode == "azure-di":
            readers[".pdf"] = azure_reader
        elif self.reader_mode == "docling":
            readers[".pdf"] = docling_reader

        dev_readers, _, _ = dev_settings()
        readers.update(dev_readers)

        return readers

    @classmethod
    def get_user_settings(cls):
        return {
            "reader_mode": {
                "name": "File loader",
                "value": "default",
                "choices": [
                    ("Default (open-source)", "default"),
                    ("Adobe API (figure+table extraction)", "adobe"),
                    (
                        "Azure AI Document Intelligence (figure+table extraction)",
                        "azure-di",
                    ),
                    ("Docling (figure+table extraction)", "docling"),
                ],
                "component": "dropdown",
            },
        }

    @classmethod
    def get_pipeline(cls, user_settings, index_settings) -> BaseFileIndexIndexing:
        use_quick_index_mode = user_settings.get("quick_index_mode", False)
        print("use_quick_index_mode", use_quick_index_mode)
        obj = cls(
            embedding=embedding_models_manager[
                index_settings.get(
                    "embedding", embedding_models_manager.get_default_name()
                )
            ],
            run_embedding_in_thread=use_quick_index_mode,
            reader_mode=user_settings.get("reader_mode", "default"),
        )
        return obj

    def is_url(self, file_path: str | Path) -> bool:
        return isinstance(file_path, str) and (
            file_path.startswith("http://") or file_path.startswith("https://")
        )

    def route(self, file_path: str | Path) -> IndexPipeline:
        """Decide the pipeline based on the file type

        Can subclass this method for a more elaborate pipeline routing strategy.
        """

        _, dev_chunk_size, dev_chunk_overlap = dev_settings()

        chunk_size = self.chunk_size or dev_chunk_size
        chunk_overlap = self.chunk_overlap or dev_chunk_overlap

        # check if file_path is a URL
        if self.is_url(file_path):
            reader = web_reader
        else:
            assert isinstance(file_path, Path)
            ext = file_path.suffix.lower()
            reader = self.readers.get(ext, unstructured)
            if reader is None:
                raise NotImplementedError(
                    f"No supported pipeline to index {file_path.name}. Please specify "
                    "the suitable pipeline for this file type in the settings."
                )

        print(f"Chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")

        print("Using reader", reader)
        pipeline: IndexPipeline = IndexPipeline(
            loader=reader,
            splitter=TokenSplitter(
                chunk_size=chunk_size or 1024,
                chunk_overlap=chunk_overlap or 256,
                separator="\n\n",
                backup_separators=["\n", ".", "\u200B"],
            ),
            run_embedding_in_thread=self.run_embedding_in_thread,
            Source=self.Source,
            Index=self.Index,
            VS=self.VS,
            DS=self.DS,
            FSPath=self.FSPath,
            user_id=self.user_id,
            private=self.private,
            embedding=self.embedding,
        )

        return pipeline

    def run(
        self, file_paths: str | Path | list[str | Path], *args, **kwargs
    ) -> tuple[list[str | None], list[str | None]]:
        raise NotImplementedError

    def stream(
        self, file_paths: str | Path | list[str | Path], reindex: bool = False, **kwargs
    ) -> Generator[
        Document, None, tuple[list[str | None], list[str | None], list[Document]]
    ]:
        """Return a list of indexed file ids, and a list of errors"""
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        file_ids: list[str | None] = []
        errors: list[str | None] = []
        all_docs = []

        n_files = len(file_paths)
        for idx, file_path in enumerate(file_paths):
            if self.is_url(file_path):
                file_name = file_path
            else:
                file_path = Path(file_path)
                file_name = file_path.name

            yield Document(
                content=f"Indexing [{idx + 1}/{n_files}]: {file_name}",
                channel="debug",
            )

            try:
                pipeline = self.route(file_path)
                file_id, docs = yield from pipeline.stream(
                    file_path, reindex=reindex, **kwargs
                )
                all_docs.extend(docs)
                file_ids.append(file_id)
                errors.append(None)
                yield Document(
                    content={
                        "file_path": file_path,
                        "file_name": file_name,
                        "status": "success",
                    },
                    channel="index",
                )
            except Exception as e:
                logger.exception(e)
                file_ids.append(None)
                errors.append(str(e))
                yield Document(
                    content={
                        "file_path": file_path,
                        "file_name": file_name,
                        "status": "failed",
                        "message": str(e),
                    },
                    channel="index",
                )

        return file_ids, errors, all_docs