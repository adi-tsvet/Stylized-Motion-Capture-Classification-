'''
Tools for Manipulating and Converting 3D Rotations

By Omid Alemi
Created: June 12, 2017

Adapted from that matlab file...
'''

import math
import numpy as np

def deg2rad(x):
    return x/180*math.pi


def rad2deg(x):
    return x/math.pi*180

class Rotation():
    def __init__(self,rot, param_type, rotation_order, **params):
        self.rotmat = []
        self.rotation_order = rotation_order
        if param_type == 'euler':
            self._from_euler(rot[0],rot[1],rot[2], params)
        elif param_type == 'expmap':
            self._from_expmap(rot[0], rot[1], rot[2], params)


    def _to_rightHand(self, ):

        M = np.asarray([[-1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

        self.rotmat = np.matmul(M, self.rotmat)
        self.rotmat = np.matmul(self.rotmat, M)
        return self.to_euler()

    def _from_euler(self, alpha, beta, gamma, params):
        '''Expecting degrees'''
        # From: https://www.gregslabaugh.net/publications/euler.pdf

        if params['from_deg']==True:
            alpha = deg2rad(alpha)
            beta = deg2rad(beta)
            gamma = deg2rad(gamma)
        
        ca = math.cos(alpha)
        cb = math.cos(beta)
        cg = math.cos(gamma)
        sa = math.sin(alpha)
        sb = math.sin(beta)
        sg = math.sin(gamma)        

        Rx = np.asarray([[1, 0, 0], 
              [0, ca, sa],
              [0, -sa, ca]
              ])

        Ry = np.asarray([[cb, 0, -sb],
              [0, 1, 0],
              [sb, 0, cb]])

        Rz = np.asarray([[cg, sg, 0],
              [-sg, cg, 0],
              [0, 0, 1]])

        self.rotmat = np.eye(3)

        ############################ inner product rotation matrix in order defined at BVH file #########################
        for axis in self.rotation_order:
            if axis == 'X' :
                self.rotmat = np.matmul(Rx, self.rotmat)
            elif axis == 'Y':
                self.rotmat = np.matmul(Ry, self.rotmat)
            else :
                self.rotmat = np.matmul(Rz, self.rotmat)
        ################################################################################################################
   
    def _from_expmap(self, alpha, beta, gamma, params):
        #From https: // github.com / una - dinosauria / human - motion - prediction / blob / master / src / data_utils.py



        if alpha == 0 and beta == 0 and gamma == 0:
            self.rotmat = np.eye(3)
            return

        #TODO: Check exp map params

        theta = np.linalg.norm([alpha, beta, gamma])

        expmap = [alpha, beta, gamma] / theta

        # r0x = np.array([0, -expmap[2], expmap[1], 0, 0, -expmap[0], 0, 0, 0]).reshape(3, 3)
        # r0x = r0x - r0x.T
        r0x = np.array([0, -expmap[2], expmap[1], expmap[2], 0, -expmap[0], -expmap[1], expmap[0], 0]).reshape(3, 3)
        self.rotmat = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * np.matmul(r0x,  r0x)


        # x = expmap[0]
        # y = expmap[1]
        # z = expmap[2]

        # s = math.sin(theta/2)
        # c = math.cos(theta/2)
        #
        # self.rotmat = np.asarray([
        #     [2*(x**2-1)*s**2+1,  2*x*y*s**2-2*z*c*s,  2*x*z*s**2+2*y*c*s],
        #     [2*x*y*s**2+2*z*c*s,  2*(y**2-1)*s**2+1,  2*y*z*s**2-2*x*c*s],
        #     [2*x*z*s**2-2*y*c*s, 2*y*z*s**2+2*x*c*s , 2*(z**2-1)*s**2+1]
        # ])



    def get_euler_axis(self):
        R = self.rotmat
        theta = math.acos((self.rotmat.trace() - 1.0) / 2.0)
        axis = np.asarray([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])


        if np.fabs(self.rotmat.trace() + 1) < 1e-12:
            theta = math.pi
            axis = 1.0 / math.sqrt(2 * (1 + R[2, 2])) * np.asarray([R[0, 2], R[1, 2], 1 + R[2, 2]])
        if np.fabs(math.sin(theta)) < 1e-12:
            # theta = np.pi
            axis = np.asarray([0, 0, 0])
            # axis = 1.0 / math.sqrt(2 * (1 + R[0, 1])) * np.asarray([R[0, 2], 1+ R[1, 1],  R[2, 1]])
        else:
            axis = axis/(2*math.sin(theta))

        return theta, axis

    def to_expmap(self):
        theta, axis = self.get_euler_axis()
        rot_arr = axis * theta
        if np.isnan(rot_arr).any():
            rot_arr = [0, 0, 0]
        return rot_arr
    
    def to_euler(self, use_deg=False):
        # From: https://www.gregslabaugh.net/publications/euler.pdf
        eulers = np.zeros((2, 3))

        if np.absolute(np.absolute(self.rotmat[0, 2]) - 1) < 1e-12:
            #GIMBAL LOCK!
            print('Gimbal')

            if np.fabs(self.rotmat[2, 0] + 1) < 1e-12:
                eulers[:, 0] = math.atan2(self.rotmat[0, 1], self.rotmat[0, 2])
                eulers[:,1] = math.pi/2
            else:
                eulers[:, 0] = math.atan2(-self.rotmat[0, 1], -self.rotmat[0, 2])
                eulers[:, 1] = -math.pi / 2

            # if np.absolute(self.rotmat[0, 2]) - 1 < 1e-12:
            #     eulers[:,0] = math.atan2(-self.rotmat[0,1], -self.rotmat[0,2])
            #     eulers[:,1] = -math.pi/2
            # else:
            #     # Funda:    eulers[:,0] = math.atan2(self.rotmat[0,1], - self.rotmat[0,2]) seklindeydi
            #     eulers[:,0] = math.atan2(self.rotmat[0,1], self.rotmat[0,2])
            #     eulers[:,1] = math.pi/2
            #
            return eulers

        # Funda: Their orders were wrong

        theta = -math.asin(self.rotmat[0,2])
        theta2 = math.pi - theta

        #x
        # psi1, psi2
        eulers[0,0] = math.atan2(self.rotmat[1,2]/math.cos(theta), self.rotmat[2,2]/math.cos(theta))
        eulers[1,0] = math.atan2(self.rotmat[1,2]/math.cos(theta2), self.rotmat[2,2]/math.cos(theta2))

        # y
        # theta1, theta2
        eulers[0,1] = theta
        eulers[1,1] = theta2

        #z
        # phi1, phi2
        eulers[0,2] = math.atan2(self.rotmat[0,1]/math.cos(theta), self.rotmat[0,0]/math.cos(theta))
        eulers[1,2] = math.atan2(self.rotmat[0,1]/math.cos(theta2), self.rotmat[0,0]/math.cos(theta2))

        if use_deg:
            eulers = rad2deg(eulers)

        return eulers
    
    def to_quat(self):
        #TODO
        pass
    
    def __str__(self):
        return "Rotation Matrix: \n " + self.rotmat.__str__()
    



