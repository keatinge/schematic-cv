import numpy as np
import matplotlib.pyplot as plt
import base64
import flask
import collections
import cv2
import time

PLOT_ARROWS = True
SHOW_GUN_PLOTS = False


def contour_color_stats(contour, image):
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)


    vals = image[mask==255]
    return {"mean" : vals.mean(), "median" : np.median(vals), "min" : vals.min(), "max" : vals.max()}



def is_gun_component(contour, im_gs, color_median):
    size = cv2.contourArea(contour)

    if size >= 70:
        return True

    if size <= 20: return False


    color = contour_color_stats(contour, im_gs)["median"]
    return color > color_median and color < 240


def contour_gun_objects(orig_image, im_gs):
    lines_removed = cv2.medianBlur(im_gs, 11)


    im_gs_white_on_black = 255 - im_gs
    eroded = cv2.erode(im_gs_white_on_black, np.ones((3,3)), iterations=1)


    med_blurred = cv2.medianBlur(eroded, 5)

    dil = cv2.dilate(med_blurred, np.ones((5,5)))


    if SHOW_GUN_PLOTS:
        plt.imshow(dil, cmap="gray")
        plt.show()


    canny = cv2.Canny(dil, 150, 300)


    im, conts, hier = cv2.findContours(med_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_color_medians = np.array([contour_color_stats(cont, im_gs)["median"] for cont in conts if cv2.contourArea(cont) < 100])
    small_color_median = np.median(small_color_medians[small_color_medians < 240])

    gun_components = [cont for cont in conts if is_gun_component(cont, im_gs, small_color_median)]
    non_components = [cont for cont in conts if not is_gun_component(cont, im_gs, small_color_median)]



    if SHOW_GUN_PLOTS:
        cv2.drawContours(orig_image, gun_components, -1, [255, 0, 0], 1)
        plt.imshow(orig_image)
        plt.show()


    return gun_components

class ArrowWindow:
    def __init__(self, window_color):
        self.window_color = window_color
        self.potential_arrows = []

    def add_arrow(self, arrow):
        self.potential_arrows.append(arrow)

class ArrowContour:
    def __init__(self, cont, window, win_x, win_y, win_rad):
        self.contour = cont
        self.absolute_contour = cont + [[win_x - win_rad, win_y - win_rad]]
        self.area = cv2.contourArea(cont)
        color_stats = contour_color_stats(cont, window)
        self.median_color = color_stats["median"]
        self.min_color = color_stats["min"]
        self.max_color = color_stats["max"]


    def get_abs_cm(self):
        m = cv2.moments(self.absolute_contour)
        x,y = m["m10"] / m["m00"], m["m01"] / m["m00"]
        return int(x),int(y)


class AnalyzedArrow:
    def __init__(self, abs_contour, seek_dv, center_pt, heavy_pt, light_pt, im_gs): #TODO REMOVE IM_GS ARGUMENT
        hx, hy = heavy_pt
        lx, ly = light_pt
        x,y = center_pt
        dx_raw = hx - lx
        dy_raw = hy - ly
        mag = ((dx_raw**2) + (dy_raw**2))**(1/2)
        dx_normed = dx_raw/mag
        dy_normed = dy_raw/mag

        assert(abs(((dx_normed**2) + (dy_normed**2))**(1/2) - 1) < 0.0000001) # Should be a unit vector

        dt = 15
        backward_dt = -3
        x0 = int(x + dx_normed * backward_dt)
        y0 = int(y + dy_normed * backward_dt)

        x1 = int(x + dx_normed * dt)
        y1 = int(y + dy_normed * dt)


        self.ray_start = (x0, y0)
        self.ray_end = (x1, y1)
        self.unit_vector = (dx_normed, dy_normed)
        self.ortho_unit_vector = (dy_normed, -dx_normed)
        self.abs_contour = abs_contour
        self.seek_dv = seek_dv
        self.center = (x,y)


        # This could technically be done in a small quadrant as relative coordinates
        # then converted to absolute coordinates but it's already pretty fast

        ray_mask = np.zeros(im_gs.shape, np.uint8)
        cv2.line(ray_mask, self.ray_start, self.ray_end, 255, 1)
        self.points_on_line = np.array(tuple(reversed(np.nonzero(ray_mask)))).transpose()

        #self.find_arrow_tail(im_gs) not complete



    def find_arrow_tail(self, im_gs):

        # THIS FUNCTION IS NOT COMPLETE!
        ortho_uv = np.array(self.ortho_unit_vector)
        arrow_head_uv = np.array(self.unit_vector)
        arrow_tail_uv = -arrow_head_uv
        center_vector = np.array(self.center)



        ortho_dist = 5
        p1 = center_vector + ortho_dist * ortho_uv
        p2 = center_vector + ortho_dist * -ortho_uv


        parr_dist = 10
        arrow_tip = center_vector + parr_dist * arrow_head_uv


        p3 = arrow_tip + ortho_dist * ortho_uv
        p4 = arrow_tip + ortho_dist * -ortho_uv

        contour = np.array([[p1.astype(int)], [p2.astype(int)], [p4.astype(int)], [p3.astype(int)]])

        cv2.drawContours(im_gs, [contour], -1, 0, 1)


        x,y = self.center





def detect_arrows(orig_image, im_gs, gun_contours):
    arrow_contours = detect_arrow_contours(orig_image, im_gs, gun_contours)

    im_gs_only_contours = im_gs.copy()

    arrow_contour_mask = np.zeros(im_gs.shape, np.uint8)
    cv2.drawContours(arrow_contour_mask, [arr.absolute_contour for arr in arrow_contours], -1, 255, -1)
    cv2.GaussianBlur(arrow_contour_mask, (5,5), 0)
    im_gs_only_contours[arrow_contour_mask != 255] =  255


    analyzed_arrows = []


    # Find the direction the arrow is pointing
    for arrow_i, arrow in enumerate(arrow_contours):
        [[vx], [vy], [x], [y]] = cv2.fitLine(arrow.absolute_contour, cv2.DIST_L2, 0, 0.1, 0.1)

        diff = 0
        dv = 6
        rad = 3

        while diff < 1/3 * (2 * rad + 1)**2: # 1/3 the area
            x0, y0 = x + dv * -vx, y + dv * -vy
            x1, y1 = x + dv * vx, y + dv * vy

            x, y = map(int, (x, y))
            x0, y0 = map(int, (x0, y0))
            x1, y1 = map(int, (x1, y1))


            #print(x0)
            #print(type(x0), type(y0), type(x1), type(y1))

            sq0 = im_gs[y0-rad:y0+rad+1, x0-rad:x0+rad+1]
            sq1 = im_gs[y1-rad:y1+rad+1, x1-rad:x1+rad+1]

            sq0_filled_pixels = sq0[sq0 != 255].__len__()
            sq1_filled_pixels = sq1[sq1 != 255].__len__()

            s0_darker = sq0_filled_pixels > sq1_filled_pixels
            diff = abs(sq0_filled_pixels - sq1_filled_pixels)

            dv += 1

        if s0_darker:
            heavy_pt = (x0, y0)
            light_pt = (x1, y1)
        if not s0_darker:
            heavy_pt = (x1, y1)
            light_pt = (x0, y0)



        cv2.drawContours(orig_image, [arrow.absolute_contour], -1, (255, 0, 0), 1)
        analyzed = AnalyzedArrow(abs_contour=arrow.absolute_contour, seek_dv=dv, center_pt=(x,y), heavy_pt=heavy_pt, light_pt=light_pt, im_gs=im_gs)
        analyzed_arrows.append(analyzed)
    return analyzed_arrows

def detect_arrow_contours(orig_image, im_gs, gun_contours):
    plt.subplots_adjust(hspace=.9)
    im_gs_cp = im_gs.copy()

    cv2.drawContours(im_gs_cp, gun_contours, -1, 255, -1)
    ret, bin = cv2.threshold(im_gs_cp, 100, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.medianBlur(bin, 5)
    im, arrow_center_contours, hi = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    windows = []
    WINDOW_RADIUS = 10

    areas = []
    median_colors = []
    min_colors = []

    for i,cnt in enumerate(arrow_center_contours):
        moments = cv2.moments(cnt)

        if moments["m00"] == 0:
            x,y = cnt[0][0]
        else:
            x,y = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

        window = im_gs_cp[y-WINDOW_RADIUS:y+WINDOW_RADIUS, x-WINDOW_RADIUS:x+WINDOW_RADIUS]
        window_color = orig_image[y-WINDOW_RADIUS:y+WINDOW_RADIUS, x-WINDOW_RADIUS:x+WINDOW_RADIUS]
        ret, window_th = cv2.threshold(window, 150, 255, cv2.THRESH_BINARY_INV)

        im, inner_arrow_contours, hi = cv2.findContours(window_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_window = ArrowWindow(window_color)


        for cnt in inner_arrow_contours:
            arr = ArrowContour(cnt, window, x, y, WINDOW_RADIUS)

            if arr.area < 10 or arr.area > 70: continue

            areas.append(arr.area)
            median_colors.append(arr.median_color)
            min_colors.append(arr.min_color)
            current_window.add_arrow(arr)

        windows.append(current_window)


    med_area = np.median(areas)
    med_col = np.median(median_colors)
    med_min_col = np.median(min_colors)


    area_low = np.percentile(areas, 10)
    area_high = np.percentile(areas, 90)
    col_high = np.percentile(median_colors, 80)


    i = 1
    best_arrows = []
    plt.suptitle("Arrow detection")
    for window in windows:
        err = lambda arrow: (arrow.area - med_area)**2 + (arrow.median_color - med_col)**2 + (arrow.min_color - med_min_col)**2

        win_col_cp = window.window_color.copy()
        pot_arrows = window.potential_arrows

        new_pot_arrows = [arr for arr in pot_arrows if abs(arr.min_color-med_min_col) <= 10]
        if new_pot_arrows:
            best_arrow = min(new_pot_arrows, key=err)
            cv2.drawContours(win_col_cp, [best_arrow.contour], -1, (0, 255, 0), 1)
            best_arrows.append(best_arrow)


        if PLOT_ARROWS:
            plt.subplot(10,10,i)
            plt.title(str(best_arrow.min_color if new_pot_arrows else "no arrow"))
            i += 1

            plt.imshow(win_col_cp)


    if PLOT_ARROWS:

        plt.show()
    return best_arrows


class ContourDetectionMask:
    def __init__(self, gun_contours, im_gs):
        self.detection_arr = np.zeros(im_gs.shape, np.uint8)
        self.contour_lookup = {}
        self.gun_contours = gun_contours

        for i, contour in enumerate(gun_contours):
            index_to_place = i + 1
            self.contour_lookup[i+1] = gun_contours[i]
            cv2.drawContours(self.detection_arr, gun_contours, i, i + 1, -1)



def pair_arrows(analyzed_arrows, contour_detection_mask, orig_image):


    cv2.drawContours(orig_image, contour_detection_mask.gun_contours, -1, (255, 0, 0), 1)

    for i,cont in enumerate(contour_detection_mask.gun_contours):
        pt = cont[0][0]
        cv2.putText(orig_image, "C" +str(i+1), tuple(pt), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)



    for i, arr in enumerate(analyzed_arrows):
        cv2.line(orig_image, arr.ray_start, arr.ray_end, (0, 255, 0), 1)
        cv2.putText(orig_image, "A"+str(i), arr.ray_start, cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


    total =0
    hits = 0
    for i, arrow in enumerate(analyzed_arrows):
        pts = arrow.points_on_line

        [x_coords, y_coords] = pts.transpose()

        all_detections = contour_detection_mask.detection_arr[y_coords,x_coords]
        real_detections = np.unique(all_detections[all_detections.nonzero()])

        total += 1
        if len(real_detections) > 0:
            hits += 1

        print("ARROW", i, "INTERSECTS", real_detections)
    plt.title("Contours and arrows")


    plt.imshow(orig_image)
    cv2.imwrite('500_contours_and_arrows.png', orig_image)
    plt.show()


def show(im):
    cv2.namedWindow("im", cv2.WINDOW_NORMAL)
    cv2.imshow("im", im)
    cv2.waitKey(0)

def arrow_testing(orig_image, im_gs, gun_contours):
    im_gs_no_gun = im_gs.copy()

    dilated_gcontour = np.zeros(im_gs.shape, np.uint8)
    cv2.drawContours(dilated_gcontour, gun_contours, -1, 255, -1) #Draw gun contours
    dilated = cv2.dilate(dilated_gcontour, np.ones((2, 2))) # Dilate them

    im_gs_no_gun[dilated == 255] = 255

    plt.imshow(im_gs_no_gun, cmap="gray")
    plt.show()

    im_gs_no_gun = 255 - im_gs_no_gun
    plt.imshow(im_gs_no_gun)



    mb = cv2.medianBlur(im_gs_no_gun, 3)



    canny = cv2.Canny(mb, 1000, 600)

    im, potential_arrow_contours, hi = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    win_size = 15


    for i, arr in enumerate(potential_arrow_contours):
        m = cv2.moments(arr)

        if m["m00"] == 0: continue



        cx, cy = (1/(m["m00"] or 1) * np.array([m["m10"], m["m01"]])).astype(int)


        win = im_gs_no_gun[cy-win_size:cy+win_size+1, cx-win_size:cx+win_size+1]


        win_rgb = cv2.cvtColor(win, cv2.COLOR_GRAY2RGB)


        edges = cv2.Canny(win, 500, 0)

        im, conts, hi = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best = max(conts, key=cv2.contourArea)

        cv2.drawContours(win_rgb, [best], -1, [0, 255, 0], 1)



        plt.subplot(15,15,i+1)
        plt.imshow(win_rgb, cmap="gray")
    plt.show()




    #plt.figure()
    #orig_image[canny > 0] = [0, 255, 0]






def build_angle_rect(start, par_vector, perp_vector, par_dist, perp_dist, start_t=4):


    assert par_dist % 2 == 1
    par_radius = (par_dist - 1) // 2


    true_start = start + perp_vector * start_t


    bottom_right = true_start + par_vector * par_radius
    bottom_left = true_start + par_vector * -par_radius

    top_right = bottom_right + perp_vector * perp_dist
    top_left = bottom_left + perp_vector * perp_dist


    return np.array([[top_left], [top_right], [bottom_right], [bottom_left]]).astype(int)





def generate_normal_vectors(orig_image, contour):


    normal_4_vecs = []

    # Add contour[-1] to calculate smoothing on the first point
    # Add contour[0] to calculate last point to first point
    # Add contour [1] to calculate smoothing for last to first
    last_fixed = np.concatenate(([contour[-1]], contour, contour[[0,1]]))


    # STEP 1: GENERATE VECTORS PERPENDICULAR TO THE CONTOUR

    for i in range(1, len(last_fixed)-1):
        prev_pt = last_fixed[i-1][0]
        this_pt = last_fixed[i][0]
        next_pt = last_fixed[i+1][0]


        prev_to_this = this_pt - prev_pt
        this_to_next = next_pt - this_pt

        prev_to_this_norm =  prev_to_this / np.linalg.norm(prev_to_this)
        this_to_next_norm = this_to_next / np.linalg.norm(this_to_next)

        smoothed_unit_vec = 1/2 * (prev_to_this_norm + this_to_next_norm)

        distance_to_next_pt = np.linalg.norm(this_to_next)


        # If there's too big of a gap, need to interpolate and add some intermediary vectors at uniform length
        if distance_to_next_pt > 3:
            pts = []
            for t in np.arange(0, distance_to_next_pt, step=3):
                pts.append(this_pt + t * this_to_next_norm)

            # If it's two far away points we can just use the exactly perpendicular vector

            perp_unit_vector = np.array([-this_to_next_norm[1], this_to_next_norm[0]])
        else:
            # They are close enough, don't interpolate
            # But because they are so close the gradients can be huge, use the average gradient in the
            # local area
            perp_unit_vector = np.array([-smoothed_unit_vec[1], smoothed_unit_vec[0]])

            pts = [this_pt]


        for point in pts:

            # Format [x, y, vx, vy]

            vec = [point[0], point[1], perp_unit_vector[0], perp_unit_vector[1]]
            normal_4_vecs.append(vec)


    normal_4_vecs.append(normal_4_vecs[0])
    final_4_vectors = []


    # In some cases the angles difference between on vec and the next can be big, insert
    # vectors in the middle to smooth the transition
    for vec, next in zip(normal_4_vecs, normal_4_vecs[1:]):
        v1_uv = [vec[2], vec[3]]
        v2_uv = [next[2], next[3]]


        dot = np.dot(v1_uv, v2_uv)
        theta = np.arccos(np.clip(np.dot(v1_uv, v2_uv), -1.0, 1.0))
        theta_deg = 180 / np.pi * theta


        if theta_deg > 30:
            # Insert vectors every 20 degrees
            for d_theta in np.arange(10, theta_deg, 10):
                d_theta_rad = d_theta * np.pi / 180
                rot_matrix = np.array([
                    [np.cos(d_theta_rad), np.sin(d_theta_rad)],
                    [-np.sin(d_theta_rad), np.cos(d_theta_rad)]
                ])
                rotated = rot_matrix.dot(vec[2:])
                final_4_vectors.append([vec[0], vec[1], rotated[0], rotated[1]])

        final_4_vectors.append(vec)


    return np.array(final_4_vectors)



def score_rect(rect_contour, score_image):
    m = cv2.moments(rect_contour)

    if m["m00"] == 0: return -1
    xcm, ycm = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


    WIDTH = 50
    roi = score_image[ycm-WIDTH:ycm+WIDTH+1, xcm-WIDTH:xcm+WIDTH+1]


    roi_mask = np.zeros(roi.shape, np.uint8)
    im_to_roi_translation_vector = np.array([xcm-WIDTH, ycm-WIDTH])
    contour_in_roi_space = rect_contour - np.array([im_to_roi_translation_vector])

    cv2.drawContours(roi_mask, [contour_in_roi_space], -1, 255, -1)

    dark_score = roi[np.logical_and(roi_mask == 255, roi < 80)].__len__()
    light_score = roi[np.logical_and(roi_mask == 255, roi < 230)].__len__() - dark_score

    #score = score_image[np.logical_and(mask == 255, score_image < 80)].__len__()
    return dark_score + .3 * light_score


def get_contour_cm(contour):
    m = cv2.moments(contour)
    xcm, ycm = 1 / m["m00"] * np.array([m["m10"], m["m01"]])

    return np.array([xcm, ycm]).astype(int)

def rotate_contour_about_cm(orig_image, contour, degrees):


    d_theta_rad = degrees * np.pi / 180
    rot_matrix = np.array([
        [np.cos(d_theta_rad), np.sin(d_theta_rad)],
        [-np.sin(d_theta_rad), np.cos(d_theta_rad)]
    ])
    xcm, ycm = get_contour_cm(contour)


    translation_to_origin = -1 * np.array([xcm, ycm])
    translation_back = -1 * translation_to_origin


    finished_points = []
    for pt in contour:
        xy_vec = pt[0]

        xy_vec_at_o = xy_vec + translation_to_origin

        xy_vec_at_o_rot = rot_matrix.dot(xy_vec_at_o)

        xy_vec_at_cm_rot = xy_vec_at_o_rot + translation_back

        finished_points.append([xy_vec_at_cm_rot])


    np_fin = np.array(finished_points).astype(int)


    return np_fin


def rotationally_maximize(orig_image, im_gs_no_gun, orig_contour):
    allowable_error = 5

    best_rect = None
    best_score = None
    best_angle = None
    for d_theta_deg in np.arange(-60, 60, allowable_error):

        new_rect = rotate_contour_about_cm(orig_image, orig_contour, d_theta_deg)

        r_score = score_rect(new_rect, im_gs_no_gun)

        if best_score is None or r_score > best_score:
            best_score = r_score
            best_rect = new_rect
            best_angle = d_theta_deg

    print(best_angle)
    assert best_rect is not None

    return best_rect, best_angle


def rotate_vector(vector, theta_in_degrees):
    d_theta_rad = theta_in_degrees * np.pi / 180

    rot_matrix = np.array([
        [np.cos(d_theta_rad), np.sin(d_theta_rad)],
        [-np.sin(d_theta_rad), np.cos(d_theta_rad)]
    ])

    return rot_matrix.dot(vector)


def find_best_rect(orig_image, im_gs_no_gun, contour):
    vecs = generate_normal_vectors(orig_image, contour)

    max_cnt = None
    max_score = None
    max_perp_v = None
    max_start = None

    break_at = None

    all_rects = []
    for i, (x, y, vx, vy) in enumerate(vecs):
        perp_unit_vector = np.array([vx, vy]) # with respect to the contour
        par_unit_vector = np.array([vy, -vx])

        rect = build_angle_rect((x,y), par_unit_vector, perp_unit_vector, par_dist=5, perp_dist=15)

        score = score_rect(rect, im_gs_no_gun)


        if max_score is None or score > max_score:
            max_score = score
            max_cnt = rect
            max_perp_v = perp_unit_vector
            max_start = x,y

            if score > 40 and not break_at:
                break_at = i + 10
       # all_rects.append(rect)

        if i == break_at:
            break


    if max_score is not None and max_score >= 10:
        rmaxed, theta = rotationally_maximize(orig_image, im_gs_no_gun, max_cnt)
        arrow_uv = -1 * rotate_vector(max_perp_v, theta)
        return rmaxed, arrow_uv
        #print(rmaxed)
    else:
        return None, None

        # UNCOMMENT TO DRAW BOXES AROUND ARROWS
        #cv2.putText(orig_image, str(max_score), tuple(max_cnt[0][0]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1,
        #            cv2.LINE_AA)
        #cv2.drawContours(orig_image, [max_cnt], -1, [0, 255, 255], 1)






def find_arrow(orig_image, im_gs, contour):
    max_cnt, max_score = find_best_rect(orig_image, im_gs, contour)

    if max_cnt is not None:
        cv2.putText(orig_image, str(max_score), tuple(max_cnt[0][0]), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        #cv2.drawContours(orig_image, [max_cnt], -1, [0, 0, 255], 3)



class GunArrowPair:
    def __init__(self, gun_contour, arrow_rect_contour, arrow_uv, orig_image):
        self.gun_contour = gun_contour
        self.arrow_rect_contour = arrow_rect_contour
        self.orig_image = orig_image
        self.arrow_uv = arrow_uv
        self.arrow_cm = get_contour_cm(arrow_rect_contour)

    def plot_both(self, color):
        cv2.line(self.orig_image, tuple(self.arrow_cm), tuple((self.arrow_cm + 10 * self.arrow_uv).astype(int)), [0, 150, 200])
        cv2.drawContours(self.orig_image, [self.arrow_rect_contour, self.gun_contour], -1, color, 1)


    def get_arrow_head_cm(self):
        moments = cv2.moments(self.arrow_rect_contour)

        return (1 / moments["m00"] * np.array([moments["m10"], moments["m01"]])).astype(int)


    def get_back_of_arrow(self, connect_components_mat):
        ax, ay = self.get_arrow_head_cm()
        width = 50
        global_to_roi_trans = -1 * np.array([ax-width, ay-width])
        roi_to_global_trans = np.array([ax-width, ay-width])

        conn_components_roi = connect_components_mat[ay - width:ay + width + 1, ax - width:ax + width + 1]


        arrow_rect_contour_in_roi = self.arrow_rect_contour + [global_to_roi_trans]


        filled_arrow_rect_contour_in_roi = np.zeros(conn_components_roi.shape, np.uint8)
        cv2.drawContours(filled_arrow_rect_contour_in_roi, [arrow_rect_contour_in_roi], -1, 255, -1)



        components_in_box = conn_components_roi[filled_arrow_rect_contour_in_roi == 255]
        components_in_box_nz = components_in_box[components_in_box.nonzero()]


        # Finds the id that occurs the most times
        chosen_id, occurences = max(zip(*np.unique(components_in_box_nz, return_counts=True)), key=lambda x: x[1])


        debug = np.zeros(conn_components_roi.shape, np.uint8)
        ys, xs = np.where(conn_components_roi == chosen_id)
        coordinates = np.array([xs, ys]).transpose()



        center_roi_space = [ax,ay] + global_to_roi_trans
        furthest = max(coordinates, key=lambda c: np.linalg.norm(c-center_roi_space))

        conn_components_roi[furthest[1], furthest[0]] = 100
        conn_components_roi[center_roi_space[1], center_roi_space[0]] = -100

        cv2.circle(self.orig_image, tuple(furthest + roi_to_global_trans), 3, [0, 100, 100], -1)
        #plt.imshow(conn_components_roi)
        #plt.show()

def arrow_method_2(orig_image, im_gs, gun_contours):


    im_gs_no_gun = im_gs.copy()
    im_gs_no_gun_dil = im_gs.copy()

    cv2.drawContours(im_gs_no_gun, gun_contours, -1, 255, -1)
    im_gs_no_gun[im_gs_no_gun > 80] = 255


    gun_contours_mask = np.zeros(im_gs.shape, np.uint8)
    cv2.drawContours(gun_contours_mask, gun_contours, -1, 255, -1)


    dil_gun_contours_mask = cv2.dilate(gun_contours_mask, np.ones((5,5)))

    im_gs_no_gun_dil[dil_gun_contours_mask > 0] = 255
    im_gs_no_gun_dil = 255 - im_gs_no_gun_dil
    im_gs_no_gun_dil[im_gs_no_gun_dil > 20] = 255
    im_gs_no_gun_dil[im_gs_no_gun_dil <= 20] = 0



    im_gs_no_gun_dil_dil = cv2.dilate(im_gs_no_gun_dil, np.ones((3,3)))
    im_gs_no_gun_dil_dil_er = cv2.erode(im_gs_no_gun_dil_dil, np.ones((3,3)))


    num_components, connect_comps_mat = cv2.connectedComponents(im_gs_no_gun_dil_dil_er, 8)

    identified_guns = []
    colors = [[0, 255, 0], [0, 0, 255]]
    for i, cnt in enumerate(gun_contours):
        best_rect, arrow_uv = find_best_rect(orig_image, im_gs_no_gun, cnt)

        if best_rect is not None:

            pair = GunArrowPair(gun_contour=cnt, arrow_rect_contour=best_rect, arrow_uv=arrow_uv, orig_image=orig_image)
            pair.plot_both(color=colors[i%len(colors)])
            pair.get_back_of_arrow(connect_comps_mat)
            identified_guns.append(pair)
        else:

            # Contours with no matching arrow
            cv2.drawContours(orig_image, [cnt], -1, [255, 0, 0], 1)
    print("IDentifiecations", len(identified_guns))

    plt.imshow(orig_image)
    plt.show()


def try_sift(orig_image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #Arrow length = 15
    #Arrow width = 6.4

    im = np.zeros((50,50), np.uint8)
    contour = np.array([
        [[0, 0]],
        [[3.2, 15]],
        [[6.4, 0]]
    ])

    im_gs = cv2.cvtColor(orig_image, cv2.COLOR_RGB2GRAY)

    contour_trans = contour + [[(50/2) - 6.4/2, (50/2) - 15/2]]
    cv2.drawContours(im, [contour_trans.astype(int)], -1, 1, -1)
    cv2.line(im, (25, 25), (25, 8), 1)


    kernel = im[8:25+15//2+1, 25-5:25+6]

    print(kernel.shape)

    plt.imshow(kernel)
    plt.show()

    kernel_height, kernel_width = kernel.shape

    kernel_x_radius = (kernel_width - 1) // 2
    kernel_y_radius = (kernel_height - 1) // 2


    conv_output = np.zeros((im_gs.shape[0] - kernel_y_radius, im_gs.shape[1] - kernel_x_radius), np.uint8)

    plt.imshow(kernel)
    plt.show()

    im, kspace_contours, hi = cv2.findContours(kernel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_contour = np.zeros(kernel.shape, np.uint8)

    cv2.drawContours(drawn_contour, kspace_contours, -1, 1, 1)
    plt.imshow(kernel)
    plt.show()

    im_height, im_width = im_gs.shape

    kernel_is_one = kernel == 1
    kernel_is_zero = kernel == 0
    plt.imshow(orig_image)
    plt.show()


    non_zero_coords = np.array(im_gs.nonzero()).transpose()
    for y,x in non_zero_coords:
        assert im_gs[y][x] != 0
        print(x, y)
        im_con_mask = im_gs[y-kernel_y_radius:y+kernel_y_radius+1, x-kernel_x_radius:x+kernel_x_radius+1]

        if im_con_mask.shape != kernel.shape:
            print("C")
            continue

        # Can this be done in one step?
        off_should_be_on = im_con_mask[np.logical_and(kernel_is_one, im_con_mask == 255)].size # Kernel is on and pixel is off
        on_should_be_off = im_con_mask[np.logical_and(kernel_is_zero, im_con_mask < 200)].size

        cv2.imshow('image', im_con_mask)
        cv2.waitKey(1)

        conv_output[y-kernel_y_radius][x-kernel_x_radius] = (kernel_height * kernel_width) - (off_should_be_on + on_should_be_off)

    min_v, max_v, min_l, max_l = cv2.minMaxLoc(conv_output)

    plt.imshow(conv_output)
    plt.show()

    cv2.circle(orig_image, max_l, 3, [0, 255, 0], 1)
    plt.imshow(orig_image)
    plt.show()


    res = cv2.matchTemplate(im_gs, template, cv2.TM_SQDIFF)
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(res)



    cv2.circle(orig_image, max_l, 5, [0, 255, 0], -1)
    plt.imshow(orig_image)
    plt.show()

    arrow = cv2.imread(r"C:\Users\Admin\Desktop\arrow.png")
    arrow_gs = cv2.cvtColor(arrow, cv2.COLOR_RGB2GRAY)[1:-2] #Image is a lil bit fucked

    arrow_kernel = arrow_gs.astype(int)

    arrow_kernel[arrow_kernel > 200] = -1
    arrow_kernel[arrow_kernel != -1] = 1


    plt.imshow(arrow_kernel)
    plt.show()




    dst = cv2.filter2D(255-im_gs, cv2.CV_64F, arrow_kernel)

    orig_image[dst > .9 * dst.max()] = [0, 255, 0]
    plt.imshow(orig_image)
    plt.show()



def process(orig_image):
    #try_other_shit(orig_image)

    im_gs = cv2.cvtColor(orig_image, cv2.COLOR_RGB2GRAY)
    gun_contours = contour_gun_objects(orig_image, im_gs)


    # Miscellenous other experiments
    # arrow_method_2(orig_image, im_gs, gun_contours)
    # arrow_testing(orig_image, im_gs, gun_contours)



    analyzed_arrows = detect_arrows(orig_image, im_gs, gun_contours)
    cdm = ContourDetectionMask(gun_contours, im_gs)
    pair_arrows(analyzed_arrows, cdm, orig_image)

    plt.imshow(orig_image)
    plt.show()



# TOdo trying measure the arrow using a literal arrow contour instead of a rectangle
# TRy rotating while using normal vectors instead of only rotating max
# Strategies to deal with arrows embeedded into contours



#process(cv2.imread(r"C:\Users\Admin\Desktop\exploded_view_500.png"))
#process(cv2.imread(r"C:\Users\Admin\Desktop\500.jpg"))
#process(cv2.imread(r"C:\Users\Admin\Desktop\835.jpg"))
process(cv2.imread(r"C:\Users\Admin\Desktop\mvp.jpg"))