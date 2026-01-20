from flow import Flow
# prototype
def loader(data):
    # input: {"id": ..., "imu_path": ..., "video_path": ...}
    # load imu and video
    # output: {"imu_data": ..., "video_data": ...}
    pass
def imu2traj(data):
    # input: {"id": ..., "imu_data": ..., "video_data": ..., "imu_path": ..., "video_path": ...}
    # process imu data
    # output: {"imu_trajectory": ...}
    pass
def video2skeleton(data):
    # input: {"id": ..., "imu_data": ..., "video_data": ..., "imu_path": ..., "video_path": ..., "imu_trajectory": ...}
    # extract skeleton from video
    # output: {"hand_trajectory": ...}, a list of dictionaries, each dictionary contains a temp person id, its trajectory, and it's border box
    pass
def matching(data):
    # input: {"id": ..., "imu_data": ..., "video_data": ..., "imu_path": ..., "video_path": ..., "imu_trajectory": ..., "hand_trajectory": ...}
    # match imu and video skeleton, find the temp person id that matches the imu data
    # output: {"person_id": ...}
    pass
def visualization(data):
    # visulaization
    pass
def evaluation(data):
    # evaluation
    pass
# example input data entry
item = {"id": "1", "imu_path": "data/imu.csv", "video_path": "data/video.mp4"}
result, err = (Flow(item)
               .bind(loader)
               .bind(imu2traj)
               .bind(video2skeleton)
               .bind(matching)
               .get())
visualization(result)
evaluation(result)