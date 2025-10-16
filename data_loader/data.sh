echo "Number of Video Files: $video_cnt"
echo "Annotation Example:"
echo ""
#!/usr/bin/env bash
# Improved dataset helper for MMAct
# - safely remove previous extractions and re-unzip
# - supports dry-run and auto-confirm
# - writes a small verify report

set -o errexit
set -o pipefail
set -o nounset

# Default data root (can be overridden with -d/--data)
DATA_ROOT="/data/lxhong/mmact_data"
LOG_FILE=""

DRY_RUN=0
AUTO_YES=0

function usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -d, --data PATH   path to data root (default: $DATA_ROOT)
  -y, --yes         auto confirm deletions (no interactive prompt)
  -n, --dry-run     show what will be done but don't delete/unzip
  -h, --help        show this help

Behavior:
  For each .zip file found in DATA_ROOT, this script will remove the
  corresponding extracted directory (basename without .zip) if it exists
  and then unzip the zip into that directory. It logs a small verify report
  to DATA_ROOT/verify_report.txt. Use --dry-run to preview actions.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data)
            DATA_ROOT="$2"
            shift 2
            ;;
        -y|--yes)
            AUTO_YES=1
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

LOG_FILE="$DATA_ROOT/verify_report.txt"

echo "=============================================="
echo "MMAct dataset helper"
echo "data root: $DATA_ROOT"
echo "$(date)"
echo "=============================================="

if ! command -v unzip >/dev/null 2>&1; then
    echo "[ERROR] unzip not found in PATH. Please install unzip."
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] data root does not exist: $DATA_ROOT"
    exit 1
fi

cd "$DATA_ROOT"

shopt -s nullglob

ZIP_FILES=("${DATA_ROOT}"/*.zip)
if [ ${#ZIP_FILES[@]} -eq 0 ]; then
    echo "No .zip files found in $DATA_ROOT"
else
    echo "Found ${#ZIP_FILES[@]} .zip files. Processing..."
fi

for z in "${ZIP_FILES[@]}"; do
    [ -f "$z" ] || continue
    bn=$(basename "$z")
    subdir="${bn%.zip}"
    echo
    echo "--- $bn -> $subdir/ ---"

    if [ -d "$subdir" ]; then
        echo "Existing directory: $subdir"
        if [ $DRY_RUN -eq 1 ]; then
            echo "DRY-RUN: would remove '$subdir'"
        else
            if [ $AUTO_YES -eq 0 ]; then
                read -r -p "Remove directory '$subdir' and re-extract? [y/N] " ans
                ans=${ans:-N}
            else
                ans=Y
            fi
            if [[ "$ans" =~ ^[Yy]$ ]]; then
                echo "Removing '$subdir'..."
                rm -rf -- "$subdir"
            else
                echo "Skipping removal of '$subdir' and skipping unzip for this archive."
                continue
            fi
        fi
    fi

    if [ $DRY_RUN -eq 1 ]; then
        echo "DRY-RUN: would unzip '$bn' -> '$subdir/'"
        continue
    fi

    echo "Unzipping '$bn' -> '$subdir/' (quiet)..."
    if unzip -q -- "$bn" -d "$subdir"; then
        echo "Unzipped: $subdir/"
    else
        echo "[ERROR] unzip failed for $bn (exit $?). Keeping any partial extraction for inspection."
    fi
done

echo
echo "Scanning key directories and counting files..."
# Use trimmed_camera* directories (your dataset uses trimmed_camera1..4)
dirs=(annotation trimmed_camera1 trimmed_camera2 trimmed_camera3 trimmed_camera4 trimmed_sensor)
for d in "${dirs[@]}"; do
    if [ -d "$d" ]; then
        echo "found: $d"
    else
        echo "missing: $d"
    fi
done

video_cnt=$(find camera* -type f -name "*.mp4" 2>/dev/null | wc -l || true)
imu_cnt=$(find trimmed_sensor -type f -name "*.csv" 2>/dev/null | wc -l || true)
anno_cnt=$(find annotation -type f -name "*.txt" 2>/dev/null | wc -l || true)

echo "videos : $video_cnt"
echo "IMU CSVs: $imu_cnt"
echo "annots : $anno_cnt"

{
    echo "========= MMAct Dataset Verify Report ========="
    date
    echo "Data path: $DATA_ROOT"
    echo ""
    echo "[dir listing]"
    ls -d */ 2>/dev/null || true
    echo ""
    echo "[counts]"
    echo "Videos: $video_cnt"
    echo "IMU CSVs: $imu_cnt"
    echo "Annotation TXT: $anno_cnt"
} > "$LOG_FILE"

echo "Report written to: $LOG_FILE"
echo "Done."