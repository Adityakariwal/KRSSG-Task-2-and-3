#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

typedef struct node {
    Point2f pt;
    int parent;
}node;

const int MAX_ITERS       = 5000;
const float STEP_SIZE     = 15.0f;
const float GOAL_RADIUS   = 10.0f;
const float SAMPLING_BIAS = 0.05f;

mt19937 rng((unsigned)time(nullptr));
uniform_real_distribution<float> uniX, uniY;


Point2f steer(const Point2f& from, const Point2f& to, float step) {
    Point2f dir = to - from;
    float dist = sqrt(dir.dot(dir));
    if (dist <= step || dist == 0)
        return to;
    return from + dir * (step / dist);
}

bool collisionFree(const Mat& img, const Point2f& a, const Point2f& b) {
    LineIterator it(img, a, b, 8);
    for(int i = 0; i < it.count; ++i, ++it) {
        Vec3b c = img.at<Vec3b>(it.pos());
        if (c[0] < 100 && c[1] < 100 && c[2] < 100)
            return false;
    }
    return true;
}

int nearestIndex(const vector<node>& tree, const Point2f& q) {
    int best = 0;
    float bestDist2 = numeric_limits<float>::max();
    for(size_t i = 0; i < tree.size(); ++i) {
        Point2f d = tree[i].pt - q;
        float d2 = d.dot(d);
        if (d2 < bestDist2) {
            bestDist2 = d2;
            best = (int)i;
        }
    }
    return best;
}

int extend(vector<node>& tree, const Point2f& q_target, const Mat& img, Mat& vis, Scalar color) {
    int idxNear = nearestIndex(tree, q_target);
    Point2f qNew = steer(tree[idxNear].pt, q_target, STEP_SIZE);
    if (!collisionFree(img, tree[idxNear].pt, qNew))
        return -1;
    node n{qNew, idxNear};
    tree.push_back(n);
    line(vis, tree[idxNear].pt, qNew, color, 1);
    imshow("RRT_Connect",vis);
    waitKey(20);
    return (int)tree.size() - 1;
}

bool connect(vector<node>& tree, const Point2f& q_target, const Mat& img, Mat& vis, Scalar color, int& lastIdx) {
    int idx = -1;
    while (true) {
        idx = extend(tree, q_target, img, vis, color);
        if (idx < 0) break;
        lastIdx = idx;
        if (norm(tree[idx].pt - q_target) <= GOAL_RADIUS)
            return true;                 
    }
    return false;
}

void drawPath(Mat& img,
              const vector<node>& treeA, int idxA,
              const vector<node>& treeB, int idxB) {
    int i = idxA;
    while (i >= 0 && treeA[i].parent != -1) {
        int p = treeA[i].parent;
        line(img, treeA[i].pt, treeA[p].pt, Scalar(0,0,255), 2);
        imshow("RRT_Connect",img);
        waitKey(20);
        i = p;
    }

    line(img, treeA[idxA].pt, treeB[idxB].pt, Scalar(0,0,255), 2);

    i = idxB;
    while (i >= 0 && treeB[i].parent != -1) {
        int p = treeB[i].parent;
        line(img, treeB[i].pt, treeB[p].pt, Scalar(0,0,255), 2);
        imshow("RRT_Connect",img);
        waitKey(20);
        i = p;
    }
}

int main(){
    Mat img = imread("maze2.png");
    if (img.empty()) return -1;
    Mat vis = img.clone();

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    Mat redMask, greenMask;
    inRange(hsv, Scalar(0,100,100), Scalar(10,255,255), redMask);
    inRange(hsv, Scalar(50,100,100), Scalar(70,255,255), greenMask);

    Point2f start,goal;
    for(int y=0; y<img.rows; y++)
    {
        for(int x=0; x<img.cols; x++)
        {
            Vec3b color = img.at<Vec3b>(Point(x, y));
            if(color[1]>150 && color[0]<100 && color[2]<100)
                start = Point2f(x,y);
            if(color[2]>200 && color[0]<100 && color[1]<100)
                goal = Point2f(x,y);
        }
    }
    cout<<start<<endl;
    cout<<goal<<endl;
    swap(start,goal);

    uniX = uniform_real_distribution<float>(0, img.cols);
    uniY = uniform_real_distribution<float>(0, img.rows);

    vector<node> treeA, treeB;
    treeA.push_back({start, -1});
    treeB.push_back({goal,  -1});

    int goalA = -1, goalB = -1;
    bool success = false;

    for(int iter=0; iter<MAX_ITERS && !success; ++iter) {
        Point2f qRand = (uniform_real_distribution<float>(0,1)(rng) < SAMPLING_BIAS)
                         ? goal : Point2f(uniX(rng), uniY(rng));

        int idxA = extend(treeA, qRand, img, vis, Scalar(255,0,0));
        if (idxA >= 0) {
            success = connect(treeB, treeA[idxA].pt, img, vis, Scalar(255,0,0), goalB);
            goalA   = idxA;
        }
        swap(treeA, treeB);
        swap(goalA, goalB);
    }

    if (success) {
        drawPath(vis, treeA, goalA, treeB, goalB);
    }

    imshow("RRT-Connect", vis);
    waitKey(0);
}
