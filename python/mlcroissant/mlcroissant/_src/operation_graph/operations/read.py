"""Read operation module."""

import dataclasses
import enum
import gzip
import io
import json
import os
import pathlib
from typing import Any

from etils import epath
import numpy as np
import pandas as pd

from mlcroissant._src.core.constants import EncodingFormat
from mlcroissant._src.core.git import download_git_lfs_file
from mlcroissant._src.core.git import is_git_lfs_file
from mlcroissant._src.core.optional import deps
from mlcroissant._src.core.path import Path
from mlcroissant._src.operation_graph.base_operation import Operation
from mlcroissant._src.operation_graph.operations.download import is_url
from mlcroissant._src.operation_graph.operations.parse_json import parse_json_content
from mlcroissant._src.structure_graph.nodes.field import Field
from mlcroissant._src.structure_graph.nodes.file_object import FileObject
from mlcroissant._src.structure_graph.nodes.file_set import FileSet
from mlcroissant._src.structure_graph.nodes.source import FileProperty

try:
    scipy = deps.scipy
except ModuleNotFoundError:
    scipy = None
INSTALL_MESSAGE = "scipy is not installed and is a dependency."


class ReadingMethod(enum.Enum):
    """Reading method derived from the fields that consume the FileObject/FileSet."""

    CONTENT = enum.auto()
    JSON = enum.auto()
    LINES = enum.auto()
    NONE = enum.auto()


def _reading_method(
    node: FileObject | FileSet, fields: tuple[Field, ...]
) -> ReadingMethod:
    """Extracts the reading method from the fields.

    If several reading methods are found, we raise an error for now. Indeed, it is
    unlikely that the same FileObject/FileSet has to be read in different manners. Also,
    an alternative solution is to define n FileObjects/FileSets when you have n
    different reading methods.
    """
    reading_methods: set[ReadingMethod] = set()
    for field in fields:
        source = field.source
        if source is None:
            continue
        extract = source.extract
        if extract.column:
            reading_methods.add(ReadingMethod.CONTENT)
        elif extract.file_property == FileProperty.lines:
            reading_methods.add(ReadingMethod.LINES)
        elif extract.file_property == FileProperty.content:
            reading_methods.add(ReadingMethod.CONTENT)
        elif extract.json_path:
            reading_methods.add(ReadingMethod.JSON)
    if len(reading_methods) == 0:
        return ReadingMethod.NONE
    if len(reading_methods) > 1:
        raise ValueError(
            f"Cannot read {node=}. The fields use several reading methods:"
            f" {reading_methods}. Reading the same FileObject/FileSet using different"
            " reading methods has yet to be implemented. Please, create an issue"
            " (https://github.com/mlcommons/croissant/issues/new) if your dataset"
            " requires this feature. Alternatively, you can use two different"
            " FileObject/FileSet pointing to the same resource."
        )
    return next(iter(reading_methods))


def _get_sampling_rate(
    node: FileObject | FileSet, fields: tuple[Field, ...]
) -> int | None:
    """Retuns the sampling rate to use for an audio file, if specified.

    If several sampling rates are used for the same audio file, an error is raised.
    """
    sampling_rates: set[int] = set()
    for field in fields:
        source = field.source
        if source is None:
            continue
        if (sr := source.sampling_rate) is not None:
            sampling_rates.add(sr)
    if len(sampling_rates) > 1:
        raise ValueError(
            f"Cannot read {node=}. The fields use several sampling rates:"
            f" {sampling_rates}. Reading the same FileObject/FileSet using different"
            " sampling rate is not possible. You can change the original sampling rate"
            " of an audio using a Transform operation."
        )
    return next(iter(sampling_rates)) if sampling_rates else None


def _should_append_line_numbers(fields: tuple[Field, ...]) -> bool:
    """Checks whether at least one field requires listing the line numbers."""
    for field in fields:
        source = field.source
        if source is None:
            continue
        if source.extract.file_property == FileProperty.lineNumbers:
            return True
    return False


def _read_arff_file(filepath: str | io.StringIO | epath.Path) -> pd.DataFrame:
    """Reads a file in ARFF format and returns it as a pandas DataFrame."""
    if scipy is None:
        raise NotImplementedError(INSTALL_MESSAGE)

    data, _ = scipy.io.arff.loadarff(filepath)
    if not isinstance(data, np.ndarray):
        raise ValueError(
            "The loaded data from scipy.io.arff does not have the expected"
            " type (a numpy array). Please ensure the ARFF file is valid."
        )
    return pd.DataFrame(data)


def _read_dicom_file(filepath: epath.Path) -> pd.DataFrame:
    """Reads a file in DICOM format and returns it as a pandas DataFrame."""
    try:
        deps.pydicom
    except ImportError as e:
        raise ImportError(
            "Missing dependency to read DICOM files. pydicom is not installed."
            " Please, install `pip install mlcroissant[dicom]`."
        ) from e
    ds = deps.pydicom.dcmread(filepath)
    pixel_array = ds.pixel_array
    return pd.DataFrame({FileProperty.content: [pixel_array]})


def _open_bioio_image(filepath: epath.Path):
    """Opens a biomedical image with bioio, falling back to Bio-Formats.

    bioio picks the most specific reader plugin for the file, so a natively
    supported file (CZI, OME-TIFF, OME-Zarr, ...) never starts the Bio-Formats
    JVM. Only when no installed plugin can read the file is the optional
    Bio-Formats reader (`mlcroissant[bioformats]`, requires Java) consulted.
    """
    try:
        bioio = deps.bioio
    except ImportError as e:
        raise ImportError(
            "Missing dependency to read biomedical image files. bioio is not"
            " installed. Please, install `pip install mlcroissant[bioio]`."
        ) from e
    path = os.fspath(filepath)
    try:
        return bioio.BioImage(path)
    except Exception as e:  # pylint: disable=broad-exception-caught
        last_error = e
    # Auto-detection keys on the file extension; nested OME-Zarr groups and
    # tar-extracted stores may not end in ".zarr", and some formats need the
    # optional Bio-Formats reader. Try those readers explicitly before giving up.
    for module in ("bioio_ome_zarr", "bioio_bioformats"):
        try:
            reader = getattr(deps, module).Reader
        except ImportError:
            continue
        try:
            return bioio.BioImage(path, reader=reader)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    raise NotImplementedError(
        f"Could not read {path!r} with the installed bioio plugins. Install a"
        " matching plugin (e.g. `pip install bioio-czi`) or the Bio-Formats fallback"
        " `pip install mlcroissant[bioformats]` (requires Java)."
    ) from last_error


def _read_bioio_file(filepath: epath.Path, fields: tuple[Field, ...]) -> pd.DataFrame:
    """Reads a biomedical image with bioio and returns it as a pandas DataFrame.

    Only image metadata (dimensions, dtype, native chunk shape) is read; the pixel
    array is materialized only when a field extracts the file `content`.
    """
    image = _open_bioio_image(filepath)
    dims = image.dims
    # Native storage chunk shape (T, C, Z, Y, X), so callers can plan chunk-aligned
    # reads with bioio (e.g. `image.get_image_dask_data(...)`) without loading pixels.
    chunk = image.dask_data.chunksize
    columns: dict[str, list[Any]] = {
        "dimension_order": [dims.order],
        "size_t": [dims.T],
        "size_c": [dims.C],
        "size_z": [dims.Z],
        "size_y": [dims.Y],
        "size_x": [dims.X],
        "chunk_t": [chunk[0]],
        "chunk_c": [chunk[1]],
        "chunk_z": [chunk[2]],
        "chunk_y": [chunk[3]],
        "chunk_x": [chunk[4]],
        "dtype": [str(image.dtype)],
    }
    df = pd.DataFrame(columns)
    extracts_content = any(
        field.source is not None
        and field.source.extract.file_property == FileProperty.content
        for field in fields
    )
    if extracts_content:
        df[FileProperty.content] = [image.data]  # type: ignore[call-overload]
    return df


@dataclasses.dataclass(frozen=True, repr=False)
class Read(Operation):
    """Reads from a file and output a pd.DataFrame."""

    node: FileObject | FileSet
    folder: epath.Path
    fields: tuple[Field, ...]

    def _read_file_content(
        self, encoding_formats: list[str], file: Path
    ) -> pd.DataFrame:
        """Extracts the `source` file to `target`."""
        filepath = file.filepath
        # bioio reads directory-based stores (e.g. OME-Zarr), so it must run before
        # the git-lfs/open("rb") checks below, which assume a single regular file.
        if EncodingFormat.BIOIO in encoding_formats:
            return _read_bioio_file(filepath, self.fields)
        if is_git_lfs_file(filepath):
            download_git_lfs_file(file)
        reading_method = _reading_method(self.node, self.fields)
        if EncodingFormat.ARFF in encoding_formats:
            return _read_arff_file(filepath)
        if EncodingFormat.DICOM in encoding_formats:
            return _read_dicom_file(filepath)

        with filepath.open("rb") as file:
            for encoding_format in encoding_formats:
                # TODO(https://github.com/mlcommons/croissant/issues/635).
                read_file: Any = file
                if filepath.suffix == ".gz":
                    read_file = gzip.open(file, "rt", newline="")
                if encoding_format == EncodingFormat.CSV:
                    return pd.read_csv(read_file)
                elif encoding_format == EncodingFormat.TSV:
                    return pd.read_csv(read_file, sep="\t")
                elif encoding_format == EncodingFormat.JSON:
                    json_content = json.load(read_file)
                    if reading_method == ReadingMethod.JSON:
                        return parse_json_content(json_content, self.fields)
                    else:
                        # Raw files are returned as a one-line pd.DataFrame.
                        return pd.DataFrame({
                            FileProperty.content: [json_content],
                        })
                elif encoding_format == EncodingFormat.JSON_LINES:
                    return pd.read_json(read_file, lines=True)
                elif encoding_format == EncodingFormat.PARQUET:
                    try:
                        df = pd.read_parquet(read_file)
                        # Sometimes the author already set an index in Parquet, so we
                        # want to reset it to always have the same format.
                        df.reset_index(inplace=True)
                        return df
                    except ImportError as e:
                        raise ImportError(
                            "Missing dependency to read Parquet files. pyarrow is not"
                            " installed. Please, install `pip install"
                            " mlcroissant[parquet]`."
                        ) from e
                elif encoding_format == EncodingFormat.TEXT:
                    if reading_method == ReadingMethod.LINES:
                        return pd.read_csv(
                            filepath, header=None, names=[FileProperty.lines]
                        )
                    else:
                        return pd.DataFrame({
                            FileProperty.content: [file.read()],
                        })
                elif encoding_format == EncodingFormat.MP3:
                    sampling_rate = _get_sampling_rate(self.node, self.fields)
                    if sampling_rate:
                        out = deps.librosa.load(file, sr=sampling_rate)
                    else:
                        out = deps.librosa.load(file)
                    return pd.DataFrame({
                        FileProperty.content: [out],
                    })
                elif encoding_format in {EncodingFormat.JPG, EncodingFormat.PNG}:
                    try:
                        img = deps.PIL_Image.open(file).convert("RGB")
                    except ModuleNotFoundError:
                        raise NotImplementedError(
                            "Missing dependency to read JPG/PNG files. Pillow is not"
                            " installed. Please, install `pip install pillow`"
                        )
                    return pd.DataFrame({FileProperty.content: [img]})
                elif encoding_format == EncodingFormat.TIF:
                    try:
                        pil_img = deps.PIL_Image.fromarray(
                            (deps.tifffile.imread(file) * 255).astype("uint8")
                        )
                    except ModuleNotFoundError:
                        raise NotImplementedError(
                            "Missing dependency to read TIF files. Pillow or tifffile"
                            " is not installed. Please, install `pip install pillow"
                            " tifffile`"
                        )
                    return pd.DataFrame({FileProperty.content: [pil_img]})
            raise ValueError(
                f"None of the provided encoding formats: {encoding_format} for file"
                f" {filepath} returned a valid pandas dataframe."
            )

    def call(self, files: list[Path] | Path) -> pd.DataFrame:
        """See class' docstring."""
        if isinstance(files, Path):
            files = [files]
        file_contents = []
        for file in files:
            # The FileObject is extracted from another FileObject/FileSet:
            if (
                isinstance(self.node, FileObject)
                and self.node.content_url
                and self.node.contained_in
            ):
                content_url = self.node.content_url
                file = Path(
                    filepath=file.filepath / content_url,
                    fullpath=pathlib.PurePath(content_url),
                )
            # The FileObject comes from disk:
            elif (
                isinstance(self.node, FileObject)
                and self.node.content_url
                and not is_url(self.node.content_url)
            ):
                # Read from the local path
                assert file.filepath.exists(), (
                    f'In node "{self.node.uuid}", file "{self.node.content_url}" is'
                    " either an invalid URL or an invalid path."
                )
            assert self.node.encoding_formats, "Encoding format is not specified."
            file_content = self._read_file_content(self.node.encoding_formats, file)
            if _should_append_line_numbers(self.fields):
                file_content[FileProperty.lineNumbers] = range(len(file_content))
            file_content[FileProperty.filepath] = file.filepath  # type: ignore[call-overload]
            file_content[FileProperty.filename] = file.filename  # type: ignore[call-overload]
            file_content[FileProperty.fullpath] = file.fullpath  # type: ignore[call-overload]
            file_contents.append(file_content)
        return pd.concat(file_contents)
