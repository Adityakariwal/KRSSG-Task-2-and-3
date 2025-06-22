
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

typedef struct node {
    Point2f p;
    int parent;
    double cost;
}node;

// Parameters
const int MAX_ITERS = 2000;
const float GOAL_RADIUS = 10.0f;
const float STEP_SIZE = 15.0f;
const float NEIGHBOR_RADIUS = 30.0f;
const double GOAL_BIAS = 0.05;

// Global random engine
typedef std::mt19937 RNGType;
RNGType rng((unsigned)time(nullptr));
uniform_real_distribution<float> uniX, uniY;

// --- Function Prototypes ---
int nearestIndex(const vector<node>& tree, const Point2f& q);
vector<int> nearIndices(const vector<node>& tree, const Point2f& q, float radius);
bool collisionFree(const Mat& img, const Point2f& a, const Point2f& b);
Point2f steer(const Point2f& from, const Point2f& to, float step);
void drawTree(Mat& img, const vector<node>& tree);
void drawPath(Mat& img, const vector<node>& tree, int idxGoal);

// --- Main ---
int main() {
    Mat img = imread("maze1.png");
    if (img.empty()) return -1;
    uniX = uniform_real_distribution<float>(0, img.cols);
    uniY = uniform_real_distribution<float>(0, img.rows);
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
    vector<node> tree;
    tree.push_back({start, -1, 0.0});
    // int bestGoalIdx=-1;

    for(int it=0; it<MAX_ITERS; ++it) {

        Point2f qrand;
        if (uniform_real_distribution<float>(0,1)(rng) < GOAL_BIAS) {
            qrand = goal;
        } else {
            qrand = Point2f(uniX(rng), uniY(rng));
        }
        

        int idx_near = nearestIndex(tree, qrand);
        Point2f qnew = steer(tree[idx_near].p, qrand, STEP_SIZE);
        

        if (!collisionFree(img, tree[idx_near].p, qnew)) continue;


        node newnode; newnode.p = qnew;
        newnode.cost = tree[idx_near].cost + norm(qnew - tree[idx_near].p);
        newnode.parent = idx_near;


        vector<int> nbrs = nearIndices(tree, qnew, NEIGHBOR_RADIUS);

        for(int i : nbrs) {
            double c = tree[i].cost + norm(tree[i].p - qnew);
            if (c < newnode.cost && collisionFree(img, tree[i].p, qnew)) {
                newnode.parent = i;
                newnode.cost = c;
            }
        }

        int newIdx = tree.size();
        tree.push_back(newnode);


        for(int i : nbrs) {
            double c_through_new = newnode.cost + norm(tree[i].p - qnew);
            if (c_through_new < tree[i].cost && collisionFree(img, qnew, tree[i].p)) {
                tree[i].parent = newIdx;
                tree[i].cost = c_through_new;
            }
        }


        line(img, tree[newIdx].p, tree[tree[newIdx].parent].p, Scalar(255,0,0));
        imshow("RRT*",img);
        waitKey(20);
        
 
        if (norm(qnew - goal) < GOAL_RADIUS) {
            double goalCost = tree[newIdx].cost + norm(goal - qnew);

            node goalNode;
            goalNode.p      = goal;
            goalNode.parent = newIdx;
            goalNode.cost   = goalCost;
            tree.push_back(goalNode);
            int goalIdx = (int)tree.size() - 1;
            drawPath(img, tree, goalIdx);
            break;

        }
    }
    imshow("RRT*", img);
    waitKey(0);
    return 0;
}

int nearestIndex(const vector<node>& tree, const Point2f& q) {
    double bestDist = numeric_limits<double>::max();
    int bestIdx = 0;
    for(size_t i=0; i<tree.size(); ++i) {
        double d = norm(tree[i].p - q);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }
    return bestIdx;
}

vector<int> nearIndices(const vector<node>& tree, const Point2f& q, float radius) {
    vector<int> ids;
    for(size_t i=0; i<tree.size(); ++i) {
        if (norm(tree[i].p - q) <= radius)
            ids.push_back(i);
    }
    return ids;
}

Point2f steer(const Point2f& from, const Point2f& to, float step) {
    Point2f dir = to - from;
    float len = sqrt(dir.x*dir.x + dir.y*dir.y);
    if (len <= step) return to;
    return Point2f(from.x + dir.x/len*step, from.y + dir.y/len*step);
}

bool collisionFree(const Mat& img, const Point2f& a, const Point2f& b) {
    LineIterator it(img, a, b, 8);
    for(int i=0; i<it.count; ++i, ++it) {
        Vec3b c = img.at<Vec3b>(it.pos());
        if (c[0]<100 && c[1]<100 && c[2]<100) return false;
    }
    return true;
}

void drawPath(Mat& img, const vector<node>& tree, int idxGoal) {
    int idx = idxGoal;
    while (tree[idx].parent != -1) {
        int p = tree[idx].parent;
        line(img, tree[idx].p, tree[p].p, Scalar(0,0,255), 2);
        idx = p;
        imshow("image",img);
        waitKey(20);
    }
}