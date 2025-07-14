import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

from decouple import config
from fsspec import AbstractFileSystem
from llama_index.readers.file.pymu_pdf import PyMuPDFReader
from PIL import Image

from kotaemon.base import Document

PDF_LOADER_DPI = config("PDF_LOADER_DPI", default=40, cast=int)


def get_page_thumbnails(
    file_path: Path, pages: list[int], dpi: int = PDF_LOADER_DPI
) -> List[Image.Image]:
    """Get image thumbnails of the pages in the PDF file.

    Args:
        file_path (Path): path to the image file
        page_number (list[int]): list of page numbers to extract

    Returns:
        list[Image.Image]: list of page thumbnails
    """

    img: Image.Image
    suffix = file_path.suffix.lower()
    assert suffix == ".pdf", "This function only supports PDF files."
    try:
        import fitz
    except ImportError:
        raise ImportError("Please install PyMuPDF: 'pip install PyMuPDF'")

    doc = fitz.open(file_path)

    output_imgs = []
    for page_number in pages:
        page = doc.load_page(page_number)
        pm = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        output_imgs.append(convert_image_to_base64(img))

    return output_imgs


def convert_image_to_base64(img: Image.Image) -> str:
    # convert the image into base64
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    img_base64 = f"data:image/png;base64,{img_base64}"

    return img_base64


class PDFThumbnailReader(PyMuPDFReader):
    """Read PDF files using PyMuPDF library."""

    def load_data(
        self,
        file: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format."""
        return self.load(file, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): file path of PDF file (accepts string or Path).
            metadata (bool, optional): if metadata to be included or not. Defaults to True.
            extra_info (Optional[Dict], optional): extra information related to each document in dict format. Defaults to None.

        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of documents.

        """
        import fitz

        # check if file_path is a string or Path
        if not isinstance(file, str) and not isinstance(file, Path):
            raise TypeError("file_path must be a string or Path.")

        # open PDF file
        doc = fitz.open(file)

        # if extra_info is not None, check if it is a dictionary
        if extra_info:
            if not isinstance(extra_info, dict):
                raise TypeError("extra_info must be a dictionary.")
            
        documents = []
        # if metadata is True, add metadata to each document
        if metadata:
            if not extra_info:
                extra_info = {}
            extra_info["file_name"] = os.path.basename(file)

            # Add documents to the list
            for page in doc:
                documents.append(
                    Document(
                        # sort=True aligns text formatting correctly
                        text=page.get_text(sort=True).encode("utf-8"),  
                        extra_info=dict(
                            extra_info,
                            **{
                                "page_label": f"{page.number + 1}",
                            },
                        ),
                    )
                )

        page_numbers_str = []
        filtered_docs = []
        is_int_page_number: dict[str, bool] = {}

        for doc in documents:
            if "page_label" in doc.metadata:
                page_num_str = doc.metadata["page_label"]
                page_numbers_str.append(page_num_str)
                try:
                    _ = int(page_num_str)
                    is_int_page_number[page_num_str] = True
                    filtered_docs.append(doc)
                except ValueError:
                    is_int_page_number[page_num_str] = False
                    continue

        documents = filtered_docs
        page_numbers = list(range(len(page_numbers_str)))

        print("Page numbers:", len(page_numbers))
        page_thumbnails = get_page_thumbnails(file, page_numbers)

        documents.extend(
            [
                Document(
                    text="Page thumbnail",
                    metadata={
                        "image_origin": page_thumbnail,
                        "type": "thumbnail",
                        "page_label": page_number,
                        **(extra_info if extra_info is not None else {}),
                    },
                )
                for (page_thumbnail, page_number) in zip(
                    page_thumbnails, page_numbers_str
                )
                if is_int_page_number[page_number]
            ]
        )

        return documents
