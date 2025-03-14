from position_utils import *
class evaluate_contour:
    def __init():
        return
    def check_closest(self,Xs,Ys,pt,tolerance = 3):
        dist = np.sqrt( (Xs-pt[0][0])**2 + (Ys-pt[0][1])**2)
        closest = np.argsort(dist)[0]
        if(dist[closest]<tolerance):
            return closest
        else:
            return False
        
    def get_points(self,contours,x,y,tolerance =3):
        points = []
        for c in contours:
            for pt in c:
                close_dot = self.check_closest(x,y,pt,tolerance)
                if(close_dot):
                    if(len(points)<0 or close_dot not in points):
                        points.append(close_dot)
        return points    
    
    def calculate_distance(self,x,y):
        p1_x = x[:-1]
        p2_x = x[1:]
        p1_y = y[:-1]
        p2_y = y[1:]
        dist = np.sqrt( (p1_x-p2_x)**2 + (p1_y-p2_y)**2)
        return dist

    def calculate_angles(self,x,y):
        angles =[]
        for i in range(len(x_sel)-1):
            angles.append(angle(x[i],y[i],x[i+1],y[i+1]))
        angles = np.array(angles)
        diff = angle_diff(angles[:-1],angles[1:])
        return diff

    def angle(self,x1,y1,x2,y2):
        angle = math.degrees(math.atan2((y2-y1), (x2-x1)))
        return angle
    
    def angle_diff(self,a1,a2):
        diff= np.abs(a1-a2)
        return diff
    
    def get_contour_score(self,line,dots,tolerance=3):
        contours = get_contours(line)
        if(len(contours)>10):
            return -3
        xs,ys = find_dot_centres(dots)
        sel_pt= self.get_points(contours,xs,ys,tolerance)
        x_sel,y_sel = xs[sel_pt],ys[sel_pt]
        dist = self.calculate_distance(x_sel,y_sel)
        positive = len(np.where(dist<15)[0])
        negative_0 = len(np.where(dist>=23)[0])
        negative_1 = len(np.where(dist>35)[0])
        #score = positive-negative_0-(2*negative_1) 
        score = positive
        return score

    