from collections import deque

class Solution:
    def findMinArrowShots(self, points) -> int:
        # sort pointers by the start point
        points = sorted(points, key=lambda p: p[0])
        num = 0
        i = 0
        while i < len(points):
            num += 1
            i += (self.get_num_intersections(i, points) + 1)
        return num
    
    def get_num_intersections(self, i, points):
        cur = points[i]
        k = 0
        for nex in points[i + 1:]:
            if nex[0] <= cur[1]:
                k += 1
        return k


if __name__ == '__main__':
    grid = [[3,9],[7,12],[3,8],[6,8],[9,10],[2,9],[0,9],[3,9],[0,6],[2,8]]
    print(Solution().findMinArrowShots(grid))