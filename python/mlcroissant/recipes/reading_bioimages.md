# Reading biomedical images with bioio

`mlcroissant` can read biomedical / microscopy images through
[bioio](https://github.com/bioio-devs/bioio). bioio is a thin reader layer over
many scientific image formats (CZI, OME-TIFF, OME-Zarr, HDF5/Imaris, ND2, LIF,
AVI, …); each format is handled by a dedicated plugin, and bioio selects the
right one from the file itself.

> For a runnable walkthrough that loads OME-TIFF, CZI and OME-Zarr and displays
> the images, see the notebook [`reading_bioimages.ipynb`](reading_bioimages.ipynb).

## What this adds

- **One umbrella encoding format.** Instead of a separate MIME type per format,
  a single `EncodingFormat.BIOIO = "application/x-bioio"` opts a `FileObject`/
  `FileSet` into the bioio reader. bioio figures out the concrete format, so new
  formats need *no* change to `mlcroissant`.
- **Metadata without pixels.** Constructing a bioio image reads only the headers
  (dimensions, dtype, native chunk shape). The pixel array is loaded **only** when
  a field extracts the file `content`. So a record set that maps only metadata
  columns never touches pixel data.
- **Bio-Formats fallback.** Files with no installed native plugin are read with
  [bioio-bioformats](https://github.com/bioio-devs/bioio-bioformats) when the
  optional `mlcroissant[bioformats]` extra (which needs a Java runtime) is
  installed.

The reader exposes these metadata columns (usable via `extract.column`):
`dimension_order`, `size_t/c/z/y/x`, `chunk_t/c/z/y/x`, `dtype`.

### Metadata-driven chunked reading

The `chunk_*` columns expose the file's **native storage chunk shape** (read from
the headers, no pixels loaded). Croissant stays a catalog/metadata layer; the
actual chunk-wise I/O is done by bioio. A typical pattern: read the `metadata`
record set to learn each image's shape and chunking, then read chunk-aligned
regions on demand with bioio:

```python
from bioio import BioImage

img = BioImage(path)                       # lazy: headers only
img.dask_data                              # chunked dask array (no pixels yet)
tile = img.get_image_dask_data(            # one chunk-aligned plane
    "YX", T=0, C=0, Z=0).compute()
```

Fine-grained partial reads are best for chunked/tiled formats (OME-Zarr, tiled
OME-TIFF, CZI mosaics); for plain contiguous files a "chunk" may be a whole plane
or stack, so `dask_data` still gives laziness/out-of-core streaming but coarser
partial I/O.

## Install from the local branch

```bash
# From a checkout of this branch:
cd python/mlcroissant
python -m venv .venv && source .venv/bin/activate

# Core library + bioio with the bundled OME-TIFF / OME-Zarr plugins:
pip install -e ".[bioio]"

# Extra plugins, installed per format as needed:
pip install bioio-czi        # .czi
pip install bioio-imageio    # .avi, .mp4, ...

# Optional Bio-Formats fallback (≈150 formats, e.g. Imaris .ims). Needs Java:
pip install -e ".[bioformats]"
```

## Example datasets

| Dataset | Format | Source | Notes |
| --- | --- | --- | --- |
| `datasets/1.1/bioio-ome-tiff` | OME-TIFF | local files in `data/` | hermetic; pixels **and** metadata-only record sets |
| `datasets/1.1/bioio-czi` | CZI | OME sample server | live (`isLiveDataset`) |
| `datasets/1.1/bioio-ome-zarr-hf` | OME-Zarr | Hugging Face | live; remote cloud store (see notebook §5) |
| `datasets/1.1/bioio-ome-zarr-cardiomyocyte` | OME-Zarr (`.zip`) | Zenodo | **loads via Croissant**: download → extract → read nested image (notebook §6) |
| `datasets/1.1/bioio-hdf5` | Imaris/HDF5 | OME sample server | live; uses Bio-Formats fallback |
| `datasets/1.1/bioio-avi` | AVI | archive.org | live |

## Run it (hermetic OME-TIFF, no network needed)

Read **only metadata** — no pixels are loaded:

```bash
mlcroissant load \
  --jsonld ../../datasets/1.1/bioio-ome-tiff/metadata.json \
  --record_set metadata --num_records 2
```

Expected output:

```text
Generating the first 2 records from ../../datasets/1.1/bioio-ome-tiff/metadata.json.
{'metadata/filename': b'synthetic_blob_0001.ome.tiff', 'metadata/dimension_order': b'TCZYX', 'metadata/size_x': 64, 'metadata/size_y': 64, 'metadata/size_c': 1, 'metadata/dtype': b'uint8', 'metadata/chunk_z': 1, 'metadata/chunk_y': 64, 'metadata/chunk_x': 64}
{'metadata/filename': b'synthetic_blob_0002.ome.tiff', 'metadata/dimension_order': b'TCZYX', 'metadata/size_x': 64, 'metadata/size_y': 64, 'metadata/size_c': 1, 'metadata/dtype': b'uint8', 'metadata/chunk_z': 1, 'metadata/chunk_y': 64, 'metadata/chunk_x': 64}
Done.
```

Read the **pixel array** (same files, `images` record set):

```bash
mlcroissant load \
  --jsonld ../../datasets/1.1/bioio-ome-tiff/metadata.json \
  --record_set images --num_records 2
```

Expected output:

```text
Generating the first 2 records from ../../datasets/1.1/bioio-ome-tiff/metadata.json.
{'images/image_filename': b'synthetic_blob_0001.ome.tiff', 'images/image_content': array([[[[[0, 0, 0, ..., 0, 0, 0],
          ...
          [0, 0, 0, ..., 0, 0, 0]]]]],
      shape=(1, 1, 1, 64, 64), dtype=uint8)}
{'images/image_filename': b'synthetic_blob_0002.ome.tiff', 'images/image_content': array([[[[[ 0,  0,  0, ...,  0,  0,  0],
          ...
          [ 0,  0,  0, ..., 15, 13, 11]]]]],
      shape=(1, 1, 1, 64, 64), dtype=uint8)}
Done.
```

The content is bioio's 5-D `(T, C, Z, Y, X)` array.

## Other formats (CZI, OME-Zarr, HDF5/Imaris, AVI)

The `bioio-{czi,ome-zarr-hf,hdf5,avi}` datasets are valid Croissant files that
*describe* real public images (run `mlcroissant validate --jsonld ...` on them);
the `bioio-ome-zarr-cardiomyocyte` dataset additionally **loads end-to-end** via
Croissant (download zip → extract → read the nested OME-Zarr). To read other
formats yourself, install the matching plugin and point bioio at **local files**
— see the notebook [`reading_bioimages.ipynb`](reading_bioimages.ipynb), which
generates an OME-Zarr store, downloads a small CZI, and loads the Cardiomyocyte
remote example.

```bash
pip install bioio-czi        # .czi
pip install bioio-imageio    # .avi, .mp4, ...
pip install -e ".[bioformats]"   # .ims / HDF5 and others (requires Java)
```

If no installed plugin can read a file you get a clear message:

```text
NotImplementedError: Could not read '.../file.czi' with the installed bioio
plugins. Install a matching plugin (e.g. `pip install bioio-czi`) or the
Bio-Formats fallback `pip install mlcroissant[bioformats]` (requires Java).
```

> **Note on remote files.** bioio chooses a reader from the file *extension*, but
> `mlcroissant`'s download cache stores a remote single file without one, so a bare
> remote `FileObject` (e.g. a `.czi` URL) is not read directly yet — use a local
> `FileSet` (as in the OME-TIFF example) or an archive. The robust pattern for
> remote data is to ship the store as a **`.zip`/`.tar`**: Croissant downloads and
> extracts it (names preserved), then a `containedIn` `FileObject` points at the
> image inside it. The reader also handles **nested OME-Zarr groups** (HCS plates,
> SpatialData) and suffix-less stores by falling back to the OME-Zarr / Bio-Formats
> readers. See the Cardiomyocyte example (notebook §6), which loads end-to-end.

## Validate any of the files (static, no download)

```bash
mlcroissant validate --jsonld ../../datasets/1.1/bioio-czi/metadata.json
# -> ... validate.py:53] Done.   (recommended-property warnings are not failures)
```

## Run the tests

```bash
# Unit tests for the reader (content, metadata-only, nested-zarr fallback,
# missing dependency):
pytest mlcroissant/_src/operation_graph/operations/read_test.py -k bioio

# Hermetic load + JSON-LD round-trip for all datasets:
pytest mlcroissant/_src/datasets_test.py -k bioio
pytest mlcroissant/_src/core/json_ld_test.py
```
