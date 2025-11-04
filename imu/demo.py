from extractor import IMUExtractor
from processor import IMUProcessor

DATA_PATH = "/data/lxhong/mmact_data"
OUTPUT_DIR = "results"
def main(
    annotation_path="annotation.json",
    data_root=DATA_PATH,
    output_dir=OUTPUT_DIR,
    sampling_rate=100,
    subject_name=None,
    init_range=(5, 30),
    zupt_threshold=0.2,
    filter_accel=False
):

    extractor = IMUExtractor(
        annotation_path=annotation_path,
        data_root=data_root
    )
    if subject_name is None: # all subjects by default
        all_data = extractor.extract_all_subjects()
        all_samples = []
        for subj_name, data_list in all_data.items():
            all_samples.extend(data_list)
    else:
        all_samples = extractor.extract_subject_data(subject_name=subject_name)
    
    print(f"{len(all_samples)} Samples in total")
    processor = IMUProcessor(
        sampling_rate=sampling_rate,
        output_dir=output_dir
    )
    results = processor.process_batch(
        data_list=all_samples,
        init_range=init_range,
        zupt_threshold=zupt_threshold,
        filter_accel=filter_accel
    )
    return results


if __name__ == "__main__":
    results = main(
        subject_name="subject11",  # None=everyone
    )