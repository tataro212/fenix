"""
models.py

This module is the single source of truth for all data structures (schemas)
used throughout the document processing pipeline.

By centralizing these Pydantic models, we enforce a strict data contract
between different parts of the application. This eliminates schema drift and
makes the entire pipeline more robust, maintainable, and easier to debug.

Every module that creates, manipulates, or consumes data about pages or
their elements MUST import the models from this file.
"""

from typing import List, Tuple, Dict, Any, Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator

# Define a type alias for a bounding box for clarity.
# Format is (x0, y0, x1, y1) representing the top-left and bottom-right corners.
BoundingBox = Tuple[float, float, float, float]

# Define a Literal type for element types to enforce a fixed set of allowed values.
ElementType = Literal["text", "image", "table", "figure", "title", "list", "caption", "quote", "footnote", "equation", "marginalia", "bibliography", "header", "footer"]

class ContentElement(BaseModel):
    """
    Represents a single content element extracted from a page.
    This model ensures consistent data structure across the pipeline.
    """
    id: str = Field(..., description="Unique identifier for this content element.")
    text: str = Field(..., description="The text content of this element.")
    label: str = Field(..., description="The type/label of this element (e.g., 'text', 'paragraph', 'title').")
    bbox: BoundingBox = Field(..., description="The bounding box of the element on the page.")
    confidence: float = Field(default=1.0, description="The confidence score from the detection model.")
    
class PageContent(BaseModel):
    """
    Represents the structured content of a single page.
    This model enforces the expected data structure between pipeline components.
    """
    page_number: int = Field(..., description="The 1-based index of the page in the document.")
    content_elements: List[ContentElement] = Field(default_factory=list, description="List of content elements found on the page.")
    image_elements: List[Dict[str, Any]] = Field(default_factory=list, description="List of image elements found on the page.")
    strategy: str = Field(default="direct_text", description="The processing strategy used for this page.")

class ElementModel(BaseModel):
    """
    Represents a single detected element on a page.

    This is a versatile model that can represent a block of text, an image,
    a table, or any other identifiable layout component. The 'type' field
    determines how the 'content' field should be interpreted.
    """
    type: ElementType = Field(..., description="The type of the element (e.g., 'text', 'image', 'table').")
    bbox: BoundingBox = Field(..., description="The bounding box of the element on the page.")
    
    # The 'content' field is a Union, allowing it to hold different data types
    # depending on the element 'type'. This provides both flexibility and type safety.
    content: Union[str, bytes, List[List[str]]] = Field(..., description="Content of the element (string for text, bytes for image, list-of-lists for table).")
    
    formatting: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of formatting attributes.")
    confidence: Optional[float] = Field(None, description="The confidence score from the detection model.")
    
    # Enhanced for Sub-Directive B: Semantic filtering support
    semantic_label: Optional[str] = Field(None, description="Semantic label detected by YOLO model (e.g., 'header', 'footer', 'paragraph') for filtering.")

class TableModel(ElementModel):
    """
    Canonical data structure for a table, inheriting from ElementModel.
    
    This model represents the structured representation of a table with 
    grid-based content, optional header row, and caption support.
    """
    type: Literal["table"] = Field(default="table", description="Element type fixed as 'table'.")
    content: List[List[str]] = Field(..., description="Grid structure representing table cells as rows x columns.")
    header_row: Optional[List[str]] = Field(None, description="Optional header row for the table.")
    caption: Optional[str] = Field(None, description="Optional caption text for the table.")

class PageModel(BaseModel):
    """
    Represents a single page of a document, using a robust validator for dimensions.

    It contains metadata about the page and a list of all elements
    found on it, each represented by an ElementModel. This is the
    "single version of truth" for a page's content and structure.
    """
    page_number: int = Field(..., description="The 1-based index of the page in the document.")
    dimensions: List[float] = Field(..., description="The (width, height) of the page. Must be exactly two floats.")
    elements: List[ElementModel] = Field(default_factory=list, description="A list of all elements detected on the page.")

    @field_validator('dimensions')
    @classmethod
    def check_dimensions_length(cls, v: List[float]) -> List[float]:
        """Ensures the dimensions list always contains exactly two items."""
        if len(v) != 2:
            raise ValueError("dimensions must be a list of exactly two floats (width, height)")
        return v

class ProcessResult(BaseModel):
    """
    A robust container for returning results from parallel worker processes.

    This standardizes the output of concurrent tasks, ensuring that errors
    are captured and handled explicitly rather than causing silent failures.
    The main process must check the 'error' field of each result.
    """
    page_number: int = Field(..., description="The page number this result corresponds to.")
    data: Optional[PageModel] = Field(None, description="The processed data, typically a PageModel object. None if an error occurred.")
    
    # Storing the error as a string is crucial for reliable serialization (pickling)
    # across process boundaries. Storing raw Exception objects can be problematic.
    error: Optional[str] = Field(None, description="A string representation of any exception that occurred during processing.")

    class Config:
        # Allows the model to handle non-Pydantic objects (like exceptions)
        # during initial creation, though we recommend converting them to strings
        # before this model is instantiated for maximum safety.
        arbitrary_types_allowed = True 