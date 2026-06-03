# Kheops Integration (DICOMweb)

## Kheops Integration (DICOMweb)

### API Client (kheops_utils.py)

```python
KHEOPS_BASE_URL = os.environ.get("KHEOPS_BASE_URL", "http://host.docker.internal")

class KheopsEndpoints:
    capabilities = f"{KHEOPS_BASE_URL}/api/capabilities"
    studies = f"{KHEOPS_BASE_URL}/api/studies"
    albums = f"{KHEOPS_BASE_URL}/api/albums"
    album_parameter = "album"
    seriesSuffix = "/series"
    instancesSuffix = "/instances"
    studyMetadataSuffix = "metadata"

def get_token_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
```

### Common API Calls

| Function | Method | URL | Purpose |
|---|---|---|---|
| `get_album_details()` | GET | `/api/albums/{album_id}` | Album metadata |
| `get_studies_from_album()` | GET | `/api/studies?album={album_id}` | List studies |
| `get_series_from_study()` | GET | `/api/studies/{study_uid}/series?album={album_id}&00080060={modality}` | Series by modality |
| `get_series_metadata()` | GET | `/api/studies/{study_uid}/series/{series_uid}/metadata?album={album_id}` | DICOM metadata |
| `download_study()` | GET | `/api/studies/{study_uid}?accept=application/zip&album={album_id}` | Download as ZIP |
| `get_user_token()` | POST | `/api/capabilities` | Create scoped capability token (24h, read+write) |

### Rules for Kheops Calls

- Always use `get_token_header(token)` for the `Authorization` header.
- For extraction tasks, create a **capability token** via `get_user_token(album_id, token)` — a time-limited scoped token.
- Handle HTTP errors explicitly — check `response.status_code` before calling `.json()`.
- DICOM field access uses hex tag codes: `DicomFields.STUDY_UID = "0020000D"`, `DicomFields.PATIENT_ID = "00100020"`, etc.
- **Never hardcode Kheops URLs** — always use `KHEOPS_BASE_URL` from environment.

---

