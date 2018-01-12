#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud_;


void pointcloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud){

        ROS_INFO("Hello!!!");
        pcl::PointCloud<pcl::PointXYZRGB> test_1_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(&test_1_);
        pcl::fromROSMsg(*cloud, test_1_);
        ROS_WARN_STREAM("Filling the vis_cloud_ using the test_1_ which has size of : " << test_1_.size());
        //vis_cloud_.reset(&test_1_);
        vis_cloud_ = tmp;
        ROS_INFO("Bye!!!");
    }

int main(int argc, char** argv){
        ros::init(argc, argv, "test_pointcloud_node");
        ros::NodeHandle nh;

        ros::Subscriber pointcloud_sup = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/sw_registered/points", 1, pointcloud_cb);


        ros::Rate my_rate(1);

        while(ros::ok()){
                ros::spinOnce();
                my_rate.sleep();
            }

        return 0;
    }
