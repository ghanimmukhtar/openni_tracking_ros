//ros addition includes
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

//original includes
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::visualization::PCLVisualizer viz;




void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_topic){
        // Fill in the cloud data
        // Generate the data
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*cloud_topic, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (0.01);

        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);

        if (inliers->indices.size () == 0)
            {
                PCL_ERROR ("Could not estimate a planar model for the given dataset.");
                return ;
            }

//        std::cerr << "Model coefficients: " << coefficients->values[0] << " "
//                  << coefficients->values[1] << " "
//                  << coefficients->values[2] << " "
//                  << coefficients->values[3] << std::endl;

        std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;

        //create a cloud that includes only the plane
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 0; i < inliers->indices.size (); ++i)
            plane_cloud->points.push_back(cloud->points[inliers->indices[i]]);


//        for (size_t i = 0; i < inliers->indices.size (); ++i)
//            std::cerr << inliers->indices[i] << "    " << cloud->points[inliers->indices[i]].x << " "
//                      << cloud->points[inliers->indices[i]].y << " "
//                      << cloud->points[inliers->indices[i]].z << std::endl;
        viz.setBackgroundColor(0, 0, 0);
        viz.setCameraClipDistances(0.00884782, 8);
        viz.setCameraPosition( 0.0926632, 0.158074, -0.955283, 0.0926631, 0.158074, -0.955282, 0.0229289, -0.994791, -0.0993251);
        viz.setCameraFieldOfView(0.7);
        viz.setSize(960, 716);
        viz.setPosition(250, 52);

        //viz.get
        viz.addPointCloud(plane_cloud, "plane_cloud");
        viz.resetCameraViewpoint("plane_cloud");
//            viz.addPlane(*coefficients);
    }

int
main (int argc, char** argv)
    {
        ros::init(argc, argv, "plannar_segmentation_node");
        ros::NodeHandle nh;

        ros::Subscriber cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/sw_registered/points", 1, cloud_cb);

        ros::Rate my_rate(10);
        while(ros::ok()){
                ros::spinOnce();
                viz.spinOnce(100, true);
                my_rate.sleep();
            }
        return (0);
    }
