import sys
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# from scipy.interpolate import griddata  # 引入scipy中的二维插值库


def img_show(SHOW_IMG, *args):
    if not SHOW_IMG: return
    n = len(args)
    for i in range(0, n, 2):
        name = args[i]
        img = args[i + 1]
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)  # 显示，第一幅图已在标准位置


def get_wraped_4_corners_range(w, h, H):
    # here x represent axis=0
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ])

    corners = np.dot(H[0], corners.T)
    corners /= corners[2, :]

    max_y, max_x, _ = corners.max(axis=1)
    min_y, min_x, _ = corners.min(axis=1)

    min_x = np.max([-h, min_x])
    min_y = np.max([-w, min_y])
    max_x = np.min([h * 2, np.max([h, max_x])])
    max_y = np.min([w * 2, np.max([w, max_y])])

    min_x = np.floor(min_x).astype(np.int32)
    min_y = np.floor(min_y).astype(np.int32)
    max_x = np.ceil(max_x).astype(np.int32)
    max_y = np.ceil(max_y).astype(np.int32)

    return min_x, min_y, max_x, max_y


# count time
def count_time_from(flag, start, info):
    cur = time.perf_counter()
    if flag: print("consume time[" + info + ": ", cur - start)
    return cur


# match
def get_good_matches(des1, des2, ratio=0.5):
    FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
    searchParams = dict(checks=50)  # 指定递归次数
    # FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器

    matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点
    good = []

    # 提取优秀的特征点
    for m, n in matches:
        if m.distance < ratio * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
            good.append(m)
    return good


def PrintError(*args):
    print("\033[1;31m", *args, "\033[0m")


def extract_kp_des(SURF, imgL_gray, imgR_gray):
    # extract
    hessian = 400  # surf
    if SURF:
        detector = cv2.xfeatures2d.SURF_create(hessian)
    else:
        detector = cv2.xfeatures2d.SIFT_create(10000)

    kp1, des1 = detector.detectAndCompute(imgL_gray, None)  # 查找关键点和描述符
    kp2, des2 = detector.detectAndCompute(imgR_gray, None)
    return kp1, kp2, des1, des2


def merge_by_homography(pano, imgR, start_i, start_j, w, h):
    imgR_location = np.asarray(
        pano[start_i:start_i + h, start_j:start_j + w])

    map = imgR_location.any(axis=2)  # np.abs( imgR_location.sum(axis=2)-imgR.sum(axis=2))/3<10000

    locations = np.where(map)
    imgR_location[locations] = imgR_location[locations] // 2 + imgR[locations] // 2
    # imgR_location.any(axis=2)
    locations = np.where(~map)
    imgR_location[locations] = imgR[locations]
    return pano


##############

# locations = np.where(imgR.any(axis=2) & pano.any(axis=2))
# imgR[locations] = pano[locations] // 2 + imgR[locations] // 2
#
# locations = np.where((~imgR.any(axis=2)) & pano.any(axis=2))
# imgR[locations] = pano[locations]
# return imgR


def get_shft(start_i, start_j):
    shft = np.array([[1.0, 0, start_j],
                     [0, 1.0, start_i],
                     [0, 0, 1.0]])
    return shft


def get_abs_offset_and_whole_size(w, h, min_x, min_y, max_x, max_y):
    i_abs_offset = max([-min_x, 0])
    j_abs_offset = max([-min_y, 0])

    whole_w = np.ceil(np.max([(max_y + j_abs_offset), w])).astype(np.int32)
    whole_h = np.ceil(np.max([(max_x + i_abs_offset), h])).astype(np.int32)

    i_abs_offset_int = np.floor(i_abs_offset).astype(np.int32)
    j_abs_offset_int = np.floor(j_abs_offset).astype(np.int32)
    return i_abs_offset_int, j_abs_offset_int, whole_w, whole_h


def get_need_depth_region(min_x, min_y, max_x, max_y):
    depth_need_w = np.floor(max_y - min_y).astype(np.int32)
    depth_need_h = np.floor(max_x - min_x).astype(np.int32)

    return depth_need_w, depth_need_h


def checkE(check, E, K, src_pts, dst_pts):
    if not check: return
    # E=np.array(
    #     [[-0.0001386345343284632, -0.0002056731812778539, -0.02894045552615692],
    # [-0.0006093625984895487, 0.02353862476458283, -0.7061227437747336],
    # [0.06840419916234093, 0.7034022269905699, 0.02334081532574019]]
    # )

    src = np.ones((src_pts.shape[0], 3))
    dst = np.ones((src_pts.shape[0], 3))
    src[:, :-1] = src_pts.reshape(src_pts.shape[0], -1)
    dst[:, :-1] = dst_pts.reshape(src_pts.shape[0], -1)
    invK = np.linalg.inv(K)

    F = np.dot(np.dot(invK.T, E), invK)
    # np.set_printoptions(threshold=sys.maxsize)
    mul = (np.dot(src, F).T * dst.T).sum(axis=0)
    print(mul)


# interpolate

def green_func(coord1, coord2):
    t1 = np.tile(coord1, (1, coord2.shape[0]))
    t2 = coord2.reshape(1, -1)

    # print("t1_shape:\n", t1.shape)
    # print("t2_shape:\n", t2.shape)
    # print("t1 :\n", t1)
    # print("t2 :\n", t2)
    # print("(t1 - t2) :\n", (t1 - t2))
    sub = (t1 - t2) ** 2

    sub = sub.reshape(-1, 2)
    sub = np.sum(sub, axis=1) ** 0.5
    # print("sub shape:\n", sub.shape)
    where = sub > 2.220446049250313e-16
    good_region = sub[where]

    sub[where] = (np.log(good_region) - 1) * good_region ** 2
    sub[~where] = 0
    return sub.reshape(coord1.shape[0], coord2.shape[0])


def get_inv_by_svd(D):
    U, s, VT = np.linalg.svd(D)

    s = np.diag(1 / s)

    S = np.zeros((D.shape[1], D.shape[0]))
    S[0:s.shape[0], 0:s.shape[0]] = s

    return VT.T.dot(S).dot(U.T)


def interpolate_v4(points_3d: np.ndarray, start_i, start_j, w, h):

    #from file
    # rootL = r'E:\dataset\ApolloScape\Depth\Record105\Camera 5'
    # nameL = r'\170908_085329643_Camera_5.png'
    # depth_img = cv2.imread(rootL + nameL)
    # coord = points_3d[:, :-1].T
    # file_depth = depth_img[coord[1], coord[0]]
    # print("file_depth shape:\n", file_depth.shape)


    # n = points_3d.shape[0]
    D = green_func(points_3d[:, :-1], points_3d[:, :-1])
    invD = get_inv_by_svd(D)
    # print("inv ok!")

    W = np.dot(invD, points_3d[:, -1])

    x = np.arange(start_j, start_j + w)
    y = np.arange(start_i, start_i + h)
    # print("mesh:\n", np.asarray(np.meshgrid(x[:3], y[:5])).reshape(2, 15).T)
    mesh = np.asarray(np.meshgrid(x, y)).reshape(2, w * h).T.astype(np.int32)

    out = np.zeros((h, w))
    for i in range(0, h):
        D1 = green_func(mesh[0 + i * w:w + i * w, :], points_3d[:, :-1])
        out[i, :] = np.dot(D1, W)
    return out.reshape(1, w * h), mesh.T


def Scatter3d(c, X, Y, Z, x0, x1, y0, y1):
    figure = plt.figure()

    ax = Axes3D(figure)

    ax.scatter(X, Y, Z, c=c)
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.show()


def get_M1_M2(R, T, from_file):
    M1 = np.zeros((3, 4))
    M2 = M1.copy()

    M1[:, 0:3] = R.T
    M1[:, 3] = -T.reshape((3,))
    M2[0:3, 0:3] = np.eye(3)
    # M1 = np.array(
    #     [
    #         [0.9984301935201793, -0.05583826449975508, -0.004385987336143508, -0.9991620374223061],
    #         [0.05595344824983785, 0.9978807054659651, 0.03321609982675056,    0.04089578329014747],
    #         [0.002521962769379205, -0.03340936809344654, 0.9994385693114842, -0.001660687565871172],
    #     ]
    # )
    arr = [
        [0.9519842070346634, -0.06932710821256463, 0.2981942682605865, 0.984867800917959],
        [0.06929053105325829, 0.9975390544598198, 0.01070780714087473, 0.102862227658016],
        [-0.2982027697103783, 0.0104683759147287, 0.9544451378904748, -0.1394803815461294]
    ]
    if from_file:
        M1 = np.array(arr)
    return M1, M2


def get_G(K1, K2, M1, M2) -> np.ndarray:
    RT1 = np.eye(4)
    RT1[:-1, :] = M1
    RT2 = np.eye(4)
    RT2[:-1, :] = M2

    Ka = np.eye(4, 4)
    Ka[0:3, 0:3] = K1
    Kb = np.eye(4, 4)
    Kb[0:3, 0:3] = K2
    # Ka_inv = numpy.linalg.inv(Ka)
    Kb_inv = np.linalg.inv(Kb)

    RT2_inv = np.linalg.inv(RT2)
    return np.linalg.multi_dot([Ka, RT1, RT2_inv, Kb_inv])

    # RT1_inv = np.linalg.inv(RT1)
    # return np.linalg.multi_dot([Ka, RT1_inv, RT2, Kb_inv])


def inv_RT(M):
    M = M.copy()
    M[:, 3] = -M[:, 3]
    M[:, :-1] = M[:, :-1].T
    return M


def get_imgL_coord(K1, K2, M1, M2, n, grid_ji, depth_map):
    G = get_G(K1, K2, M1, M2)
    mesh_homo = np.ones((4, n))
    mesh_homo[:-2, :] = grid_ji
    mesh_homo[:-1, :] = depth_map * mesh_homo[:-1, :]

    imgL_coord = G.dot(mesh_homo)
    imgL_coord /= imgL_coord[2]
    return imgL_coord[:-2]


def get_imgPano_coord(K1, K2, M1, M2, n, shft, grid_ji, depth_map):
    G = get_G(K1, K2, M1, M2)
    # G=get_inv_by_svd(G)
    G = np.linalg.inv(G)

    mesh_homo = np.ones((4, n))
    mesh_homo[:-2, :] = grid_ji
    mesh_homo[:-1, :] = depth_map * mesh_homo[:-1, :]

    imgR_coord = G.dot(mesh_homo)
    imgR_coord /= imgR_coord[2]
    imgR_coord[0, :] += shft[0, 2]
    imgR_coord[1, :] += shft[1, 2]
    return imgR_coord[:-2]


def get_imgL_coord_yx(grid_ij, n, K, M1, depth_map):  # 3*n

    mesh_homo = np.ones((3, n))
    mesh_homo[:-1, :] = grid_ij

    P_homo = np.ones((4, n))

    invK = np.linalg.inv(K)
    P_homo[:-1, :] = depth_map * invK.dot(mesh_homo)

    imgL_homo_coord = K.dot(M1).dot(P_homo)
    imgL_homo_coord /= imgL_homo_coord[-1, :]
    return imgL_homo_coord[:-1, :]


def get_construct_3d(dst_pts, depth, from_file):  # n*2,n*1->n*3
    if not from_file:
        construct_3d = np.zeros((dst_pts.shape[0], 3))
        dst_pts = dst_pts.reshape(-1, 2)
        construct_3d[:, :-1] = np.asarray([dst_pts[:, 0], dst_pts[:, 1]]).T
        construct_3d[:, -1] = depth
        return construct_3d, dst_pts

    with open("meshes.txt", "rt") as f:
        s = f.read()
        construct_3d = np.asarray(eval(s))
    out = construct_3d.reshape(construct_3d.shape[0] // 3, 3)
    return construct_3d.reshape(construct_3d.shape[0] // 3, 3), out[:, :-1]


    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > 120:
                cv2.circle(image, (j, i), 2, (0, 255, 0), 2)
    # output
    return image

####################################################
def to_numpy_int_floor(arr):
    return np.floor(arr).astype(np.int32)


def to_numpy_int_round(arr):
    return np.round(arr).astype(np.int32)


def to_numpy_uint8_round(arr):
    return np.round(arr).astype(np.uint8)


def to_numpy_int_ceil(arr):
    return np.ceil(arr).astype(np.int32)



