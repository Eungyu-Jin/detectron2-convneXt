import cv2
import numpy as np

def load_image(path: str, retrun_frames: bool=False) -> np.ndarray:
    """
    ## load_image
    이미지 파일을 RGB 형태의 numpy array로 read

    ### args
    - return_frames
        - True: gif의 모든 프레임을 list에 return 
        - False: 첫 번째 프레임만 return
    """
    if path.endswith('.gif'):
        gif = cv2.VideoCapture(path)
        ret, frame = gif.read()
        if retrun_frames:
          frames = [frame]
          while ret:
              ret, frame = gif.read()
              if ret:
                  frames.append(frame)
          return [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in frames] # 순차로 list 저장
        else:
          if ret:
              return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) # BGR -> RGB로 변환
    