from model import *

__all__ = ['build_detector', 'build_tracker', 'build_mesh_generater', 'build_face_ReID', 'build_pose_generator']

def build_detector(args):
    return YOLOFace(args)

def build_face_ReID(model_name, args=None):
    if model_name=='FaceEmbedderDirect':
        return FaceEmbedderDirect(args=args)
    elif model_name=='FaceEmbedderFR':
        return FaceEmbedderFR(args=args)
    elif model_name=='FaceEmbedderFacenet':
        return FaceEmbedderFacenet(args=args)
    elif model_name=='FaceEmbedderInsightFace':
        return FaceEmbedderInsightFace(args=args)
    elif model_name.startswith('FaceEmbedderDeepFace'):
        if model_name == 'FaceEmbedderDeepFace':
            return FaceEmbedderDeepFace(args=args)
        backend_name = model_name.split('_', 1)[1]  # 取 '_' 后面的部分
        return FaceEmbedderDeepFace(model_name=backend_name, args=args)

def build_tracker(args, frame_rate=30):
    return BoTSORT(args, frame_rate=30)

def build_mesh_generater():
    return Face3DMeshGenerator()

def build_pose_generator(args):
    return YOLOPose(args)