"""After running get_frames.py, the script reads the pickle file and
stores its contents in an array. This data will then be used for both
plotting and analyzing purposes."""


import static_info as stinfo


class ReadCom:
    """reading the center of mass file, the name is set static_info.py
    """
    def __init__(self) -> None:
        self.f_name: str = stinfo.files['com_pickle']
        print(self.f_name)


if __name__ == '__main__':
    data = ReadCom()
