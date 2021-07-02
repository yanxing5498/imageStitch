from fundamental_funcs import *
import png

start_time = time.perf_counter()
begin_time = start_time
### options ###
SHOW_IMG = 1  # show image
SHOW_STR = 1  # print in console
PRINT_MAT = 0
PRINT_CONSUME = 1
PANO = 1  # show pano_image

FROM_FILE = 0
WRITE = 1  # save pano_image to file 'pano.jpg'
SHOW_WARNING = 1
DRAW_MATCH = 1
TESTING = 0
PRINT_ALL_NUMBER = 0
SHOW_SCIENTIFIC_NOTATION = 0
WRITE_PATH = r'output/'

LEFT_ORIENTED = 1
SURF = 1
RATIO = 0.6  # define knn good matches ratio
# path1 = r"C:\Users\Mr.Yan\Desktop\pairs\5.jpg"  #_P1010520.JPG"
# path2 = r"C:\Users\Mr.Yan\Desktop\pairs\4.jpg"  #_P1010517.JPG"


#
path1 = r'C:\Users\Mr.Yan\Desktop\crazyhorse\P1000966.JPG'
path2 = r'C:\Users\Mr.Yan\Desktop\crazyhorse\P1000965.JPG'

path1 = r"output\img1_remap.bmp"  #_P1010520.JPG"
path2 = r"output\img2_remap.bmp"  #_P1010517.JPG"

if PRINT_ALL_NUMBER:
    np.set_printoptions(threshold=sys.maxsize)
if not SHOW_SCIENTIFIC_NOTATION:
    np.set_printoptions(suppress=True)

# ************************************************************************************************** #
def Print(*args):
    if SHOW_STR:
        print(*args)


def PrintMat(*args):
    if not PRINT_MAT: return
    if SHOW_STR:
        print(*args)


imgL = cv2.imread(path1)
imgR = cv2.imread(path2)
h, w = imgL.shape[:2]
h1, w1 = imgR.shape[:2]
Print("w,h: ", w, h)

imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
####################################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "pre_process")

img_show(SHOW_IMG, "imgL", imgL, "imgR", imgR)

with open('config/calibrate.txt', 'rt') as f:
    s = f.read()
    K = eval(s)

kp1, kp2, des1, des2 = extract_kp_des(SURF, imgL, imgR)  # surf or sift
Print("key_points_size: ", len(kp1), len(kp2))

mask = []
K = np.array(K)
K[0, 2] = w / 2
K[1, 2] = h / 2
Print("K:\n", K)
####################################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "extract")
good = get_good_matches(des1, des2, RATIO)
Print("goo_matches_size:\n", len(good))
####################################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "match")

# findTransform
src_pts = np.array([[kp1[m.queryIdx].pt] for m in good])  # 查询图像的特征描述子索引
dst_pts = np.array([[kp2[m.trainIdx].pt] for m in good])  # 训练(模板)图像的特征描述子索引

Print("src_pts_shape", src_pts.shape)

if SHOW_IMG or DRAW_MATCH:
    img_matches = cv2.drawMatches(imgL, kp1, imgR, kp2, good, None)
    img_show(1, "matches", img_matches)
    if WRITE:
        imgL_and_kp1=np.array(imgL)
        cv2.drawKeypoints(imgL,kp1,imgL_and_kp1)
        cv2.imwrite(WRITE_PATH + "corners.jpg", imgL_and_kp1)
        cv2.imwrite(WRITE_PATH + "matches.jpg", img_matches)

H = cv2.findHomography(src_pts, dst_pts)  # 生成变换矩阵

################################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "findE")
E, mask = cv2.findEssentialMat(src_pts, dst_pts, K[0, 0], (w / 2, h / 2),
                               method=cv2.RANSAC, prob=0.999, threshold=1)
Print("E:\n", E)
interior = np.count_nonzero(mask)

if SHOW_WARNING and interior / src_pts.shape[0] < 0.7:
    PrintError("警告：内点占比过少！")
Print("findEssentialMat interior: ", np.count_nonzero(mask), " in ", src_pts.shape)
################################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "findH")
# E=np.array(
#     [[-0.0001386345343284632, -0.0002056731812778539, -0.02894045552615692],
# [-0.0006093625984895487, 0.02353862476458283, -0.7061227437747336],
# [0.06840419916234093, 0.7034022269905699, 0.02334081532574019]]
# )
retval, R, T, mask = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
Print("R,T:\n", R, "\n", T)

M1, M2 = get_M1_M2(R, T, FROM_FILE)
################################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "findR_T")

points_3d = cv2.triangulatePoints(np.dot(K, M1), np.dot(K, M2), src_pts, dst_pts)

src_pts = src_pts.reshape(-1, 2)
dst_pts = dst_pts.reshape(-1, 2)

points_3d[:] /= points_3d[3]
Print("p3d_shape:", points_3d.shape)
Print("points_3d:\n", points_3d.T)
where = ((points_3d[2] >= 0))
Print("depth non_zero:\n", np.count_nonzero(where))
Print("where shape:\n", where.shape)

points_3d = points_3d[:, where]
src_pts = src_pts[where, :]
dst_pts = dst_pts[where, :]

# wrap
Print("H", H[0])

min_x, min_y, max_x, max_y = get_wraped_4_corners_range(w, h, H)

Print("wraped_range:\n", min_x, min_y, max_x, max_y)

i_abs_offset, j_abs_offset, whole_w, whole_h = get_abs_offset_and_whole_size(
    w, h, min_x, min_y, max_x, max_y)

depth_need_w, depth_need_h = get_need_depth_region(min_x, min_y, max_x, max_y)

Print("whole_w,whole_h: ", whole_w, whole_h)
Print("depth_need_w, depth_need_h: \n",
      depth_need_w, depth_need_h)

shft = get_shft(i_abs_offset, j_abs_offset)

Print("shft:\n", shft)

M = np.dot(shft, H[0])  # 获取左边图像到右边图像的投影映射关系

pano = cv2.warpPerspective(imgL, M, (whole_w, whole_h))  # 透视变换，新图像可容纳完整的两幅图
imgR_shft = cv2.warpPerspective(imgR, shft, (whole_w, whole_h))
if WRITE:
    cv2.imwrite(WRITE_PATH+"img1.jpg", pano)
    cv2.imwrite(WRITE_PATH+"img2.jpg", imgR_shft)
##############################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "wrap_by_homography")

Print("interpolating...")

if LEFT_ORIENTED:

    depth_true_w, depth_true_h = depth_need_w, depth_need_h
    construct_3d, dst_pts = get_construct_3d(dst_pts, points_3d.T[:, 2], FROM_FILE)
    if WRITE:
        np.savetxt(WRITE_PATH+"meshes1.txt", construct_3d)

    depth_map, grid_ji = interpolate_v4(construct_3d,
                                        min_x, min_y, depth_need_w, depth_need_h)

    Print("construct_3d shape:\n", construct_3d.shape)
    depth_img = to_numpy_uint8_round(np.clip(depth_map, 0, 255)).reshape((depth_need_h, depth_need_w))
    cv2.namedWindow("depth_img", cv2.WINDOW_NORMAL)
    cv2.imshow("depth_img", depth_img)
    if WRITE:
        cv2.imwrite(WRITE_PATH+"depth.png", depth_img)

    grid_ij = np.asarray([grid_ji[1], grid_ji[0]])

    Print("depth_map shape:\n", depth_map.shape)

    # imgL_coord = get_imgL_coord_yx(grid_ij, depth_need_w * depth_need_h, K, M1, depth_map)
    imgL_coord = get_imgL_coord(K, K, M1, M2, depth_need_w * depth_need_h, grid_ji, depth_map)

    imgL_coord_i = np.round(imgL_coord[1, :]).astype(np.int32)
    imgL_coord_j = np.round(imgL_coord[0, :]).astype(np.int32)
    imgL_coord_ij = [imgL_coord_i, imgL_coord_j]
    pano_by_depth = np.zeros_like(pano)

    imgL_coord = np.array([imgL_coord_i, imgL_coord_j])
    pano_coord = np.array([grid_ij[0] + i_abs_offset, grid_ij[1] + j_abs_offset])

    lower = (0 <= imgL_coord)
    where = ((imgL_coord[0] <= h - 1) & (imgL_coord[1] <= w - 1)) & lower.all(axis=0)

    final_pano_coord_i = pano_coord[0, where]
    final_pano_coord_j = pano_coord[1, where]
    final_imgL_coord_i = imgL_coord[0, where]
    final_imgL_coord_j = imgL_coord[1, where]

    pano_by_depth[final_pano_coord_i, final_pano_coord_j] = \
        imgL[final_imgL_coord_i, final_imgL_coord_j]

    pano_by_depth_no_merge = pano_by_depth
    pano_by_depth = pano_by_depth // 2 + imgR_shft // 2

else:

    depth_true_w, depth_true_h = w, h
    construct_3d, src_pts = get_construct_3d(src_pts, points_3d.T[:, 2], FROM_FILE)

    if WRITE:
        np.savetxt(WRITE_PATH+"meshes1.txt", construct_3d)
    depth_map, grid_ji = interpolate_v4(construct_3d,
                                        0, 0, w, h)
    Print("construct_3d shape:\n", construct_3d.shape)
    depth_img = to_numpy_uint8_round(np.clip(depth_map, 0, 255)).reshape((h, w))
    cv2.namedWindow("depth_img", cv2.WINDOW_NORMAL)
    cv2.imshow("depth_img", depth_img)
    if WRITE:
        cv2.imwrite(WRITE_PATH+"depth.png", depth_img)

    grid_ij = np.asarray([grid_ji[1], grid_ji[0]])

    Print("depth_map shape:\n", depth_map.shape)
    ##############################################################################################
    start_time = count_time_from(PRINT_CONSUME, start_time, "interp")

    # imgL_coord = get_imgL_coord(K, K, M1, M2, depth_need_w * depth_need_h, grid_ji, depth_map)
    imgPano_coord = get_imgPano_coord(K, K, M1, M2, w * h, shft, grid_ji, depth_map)

    imgPano_coord_i = np.round(imgPano_coord[1, :]).astype(np.int32)
    imgPano_coord_j = np.round(imgPano_coord[0, :]).astype(np.int32)

    pano_by_depth = np.zeros_like(pano)
    imgPano_coord = np.array([imgPano_coord_i, imgPano_coord_j])
    lower = 0 <= imgPano_coord
    where = ((imgPano_coord[0] <= whole_h - 1) & (imgPano_coord[1] <= whole_w - 1)) & lower.all(axis=0)

    final_pano_coord_i = imgPano_coord[0, where]
    final_pano_coord_j = imgPano_coord[1, where]
    final_imgL_coord_i = grid_ij[0, where]
    final_imgL_coord_j = grid_ij[1, where]
    pano_by_depth[final_pano_coord_i, final_pano_coord_j] = imgL[final_imgL_coord_i, final_imgL_coord_j]

    pano_by_depth_no_merge = pano_by_depth
    pano_by_depth = pano_by_depth // 2 + imgR_shft // 2


##############################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "interp")
img_show(SHOW_IMG or PANO, "pano_by_depth_no_merge", pano_by_depth_no_merge)
img_show(SHOW_IMG or PANO, "pano_by_depth", pano_by_depth)

# Print("depth_map:\n",depth_map)

##############################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "wrap_by_depth")

img_show(SHOW_IMG, 'imgR_shft', imgR_shft, 'pano_beforeMerge', pano)

pano = merge_by_homography(pano, imgR, i_abs_offset, j_abs_offset, w, h)

##########################################################################################
start_time = count_time_from(PRINT_CONSUME, start_time, "merge")

img_show(SHOW_IMG or PANO, 'pano', pano)

if WRITE:
    cv2.imwrite(WRITE_PATH+"pano.jpg", pano)
    cv2.imwrite(WRITE_PATH+"pano_by_depth.jpg", pano_by_depth)

print("consume time[global homography: ", start_time - begin_time, "\n")

checkE(TESTING, E, K, src_pts, dst_pts)

if not FROM_FILE:
    if LEFT_ORIENTED:
        dst_pts = src_pts

    X = dst_pts.reshape((-1, 2))[:, 0]
    Z = dst_pts.reshape((-1, 2))[:, 1]
    Y = points_3d[2, ...]
    Scatter3d("r", X, Y, Z, 0, w, 0, 100)

    depth_mat = depth_map.reshape(depth_true_h, depth_true_w)

    dst_pts_shft = to_numpy_int_floor(dst_pts + np.array([shft[0, 2], shft[1, 2]]) -
                                      np.array([j_abs_offset, i_abs_offset])
                                      )
    dst_pts_shft_T = dst_pts_shft.T
    predict_depth = depth_mat[dst_pts_shft_T[1], dst_pts_shft_T[0]].reshape(-1)
    errors = predict_depth - points_3d[2, :]
    if WRITE:
        np.savetxt(WRITE_PATH+"fit_depth.txt", predict_depth)
        np.savetxt(WRITE_PATH+"predict_depth.txt", points_3d[2, :])
        np.savetxt(WRITE_PATH+"depth_fit_errors.txt", errors)

    Z = dst_pts_shft_T[1]
    X = dst_pts_shft_T[0]
    Y = predict_depth
    Scatter3d("b", X, Y, Z, 0, w, 0, 100)
cv2.waitKey(0)
