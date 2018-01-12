/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
//ros stuff
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/io.h>
#include <chrono>

#define PCL_NO_PRECOMPILE
#define PCL_TRACKING_NORMAL_SUPPORTED

//original includes
#include <pcl/tracking/tracking.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>

#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/normal_coherence.h>

#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <pcl/common/centroid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/surface/convex_hull.h>

#include <pcl/search/pcl_search.h>
#include <pcl/common/transforms.h>

#include <boost/format.hpp>

#define FPS_CALC_BEGIN                          \
    static double duration = 0;                 \
    double start_time = pcl::getTime ();        \

#define FPS_CALC_END(_WHAT_)                    \
{                                             \
    double end_time = pcl::getTime ();          \
    static unsigned count = 0;                  \
    if (++count == 10)                          \
{                                           \
    std::cout << "Average framerate("<< _WHAT_ << "): " << double(count)/double(duration) << " Hz" <<  std::endl; \
    count = 0;                                                        \
    duration = 0.0;                                                   \
    }                                           \
    else                                        \
{                                           \
    duration += end_time - start_time;        \
    }                                           \
    }

using namespace pcl::tracking;

template <typename PointType>
class OpenNISegmentTracking
    {
    public:
        typedef pcl::PointXYZRGBNormal RefPointType;
        //typedef pcl::PointXYZRGBANormal RefPointType;
        //typedef pcl::PointXYZRGBA RefPointType;
        //typedef pcl::PointXYZ RefPointType;
        typedef ParticleXYZRPY ParticleT;

        typedef pcl::PointCloud<PointType> Cloud;
        typedef pcl::PointCloud<RefPointType> RefCloud;
        typedef typename RefCloud::Ptr RefCloudPtr;
        typedef typename RefCloud::ConstPtr RefCloudConstPtr;
        typedef typename Cloud::Ptr CloudPtr;
        typedef typename Cloud::ConstPtr CloudConstPtr;
        //typedef KLDAdaptiveParticleFilterTracker<RefPointType, ParticleT> ParticleFilter;
        //typedef KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> ParticleFilter;
        //typedef ParticleFilterOMPTracker<RefPointType, ParticleT> ParticleFilter;
        typedef ParticleFilterTracker<RefPointType, ParticleT> ParticleFilter;
        typedef typename ParticleFilter::CoherencePtr CoherencePtr;
        typedef typename pcl::search::KdTree<PointType> KdTree;
        typedef typename KdTree::Ptr KdTreePtr;
        OpenNISegmentTracking (int thread_nr, double downsampling_grid_size,
                               bool use_convex_hull,
                               bool visualize_non_downsample, bool visualize_particles,
                               bool use_fixed, bool print_time)
            : viewer_ ("PCL OpenNI Tracking Viewer")
            , new_cloud_ (false)
            , ne_ (thread_nr)
            , counter_ (0)
            , use_convex_hull_ (use_convex_hull)
            , visualize_non_downsample_ (visualize_non_downsample)
            , visualize_particles_ (visualize_particles)
            , print_time_ (print_time)
            , downsampling_grid_size_ (downsampling_grid_size)
            {
                KdTreePtr tree (new KdTree (false));
                ne_.setSearchMethod (tree);
                ne_.setRadiusSearch (0.03);

                std::vector<double> default_step_covariance = std::vector<double> (6, 0.015 * 0.015);
                default_step_covariance[3] *= 40.0;
                default_step_covariance[4] *= 40.0;
                default_step_covariance[5] *= 40.0;

                std::vector<double> initial_noise_covariance = std::vector<double> (6, 0.00001);
                std::vector<double> default_initial_mean = std::vector<double> (6, 0.0);
                if (use_fixed)
                    {
                        boost::shared_ptr<ParticleFilterOMPTracker<RefPointType, ParticleT> > tracker
                                (new ParticleFilterOMPTracker<RefPointType, ParticleT> (thread_nr));
                        tracker_ = tracker;
                    }
                else
                    {
                        boost::shared_ptr<KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> > tracker
                                (new KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> (thread_nr));
                        tracker->setMaximumParticleNum (500);
                        tracker->setDelta (0.99);
                        tracker->setEpsilon (0.2);
                        ParticleT bin_size;
                        bin_size.x = 0.1f;
                        bin_size.y = 0.1f;
                        bin_size.z = 0.1f;
                        bin_size.roll = 0.1f;
                        bin_size.pitch = 0.1f;
                        bin_size.yaw = 0.1f;
                        tracker->setBinSize (bin_size);
                        tracker_ = tracker;
                    }

                tracker_->setTrans (Eigen::Affine3f::Identity ());
                tracker_->setStepNoiseCovariance (default_step_covariance);
                tracker_->setInitialNoiseCovariance (initial_noise_covariance);
                tracker_->setInitialNoiseMean (default_initial_mean);
                tracker_->setIterationNum (1);

                tracker_->setParticleNum (400);
                tracker_->setResampleLikelihoodThr(0.00);
                tracker_->setUseNormal (true);
                // setup coherences
                ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr coherence = ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr
                        (new ApproxNearestPairPointCloudCoherence<RefPointType> ());
                // NearestPairPointCloudCoherence<RefPointType>::Ptr coherence = NearestPairPointCloudCoherence<RefPointType>::Ptr
                //   (new NearestPairPointCloudCoherence<RefPointType> ());

                boost::shared_ptr<DistanceCoherence<RefPointType> > distance_coherence
                        = boost::shared_ptr<DistanceCoherence<RefPointType> > (new DistanceCoherence<RefPointType> ());
                coherence->addPointCoherence (distance_coherence);

                boost::shared_ptr<HSVColorCoherence<RefPointType> > color_coherence
                        = boost::shared_ptr<HSVColorCoherence<RefPointType> > (new HSVColorCoherence<RefPointType> ());
                color_coherence->setWeight (0.1);
                coherence->addPointCoherence (color_coherence);

                //boost::shared_ptr<pcl::search::KdTree<RefPointType> > search (new pcl::search::KdTree<RefPointType> (false));
                boost::shared_ptr<pcl::search::Octree<RefPointType> > search (new pcl::search::Octree<RefPointType> (0.01));
                //boost::shared_ptr<pcl::search::OrganizedNeighbor<RefPointType> > search (new pcl::search::OrganizedNeighbor<RefPointType>);
                coherence->setSearchMethod (search);
                coherence->setMaximumDistance (0.01);
                tracker_->setCloudCoherence (coherence);
            }

        bool
        drawParticles (pcl::visualization::PCLVisualizer& viz)
            {
                ParticleFilter::PointCloudStatePtr particles = tracker_->getParticles ();
                if (particles)
                    {
                        if (visualize_particles_)
                            {
                                pcl::PointCloud<pcl::PointXYZ>::Ptr particle_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
                                for (size_t i = 0; i < particles->points.size (); i++)
                                    {
                                        pcl::PointXYZ point;

                                        point.x = particles->points[i].x;
                                        point.y = particles->points[i].y;
                                        point.z = particles->points[i].z;
                                        particle_cloud->points.push_back (point);
                                    }

                                {
                                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue_color (particle_cloud, 250, 99, 71);
                                    if (!viz.updatePointCloud (particle_cloud, blue_color, "particle cloud"))
                                        viz.addPointCloud (particle_cloud, blue_color, "particle cloud");
                                }
                            }
                        return true;
                    }
                else
                    {
                        PCL_WARN ("no particles\n");
                        return false;
                    }
            }

        void
        drawResult (pcl::visualization::PCLVisualizer& viz)
            {
                ParticleXYZRPY result = tracker_->getResult ();
                Eigen::Affine3f transformation = tracker_->toEigenMatrix (result);
                // move a little bit for better visualization
                transformation.translation () += Eigen::Vector3f (0.0f, 0.0f, -0.005f);
                RefCloudPtr result_cloud (new RefCloud ());

                if (!visualize_non_downsample_)
                    pcl::transformPointCloud<RefPointType> (*(tracker_->getReferenceCloud ()), *result_cloud, transformation);
                else
                    pcl::transformPointCloud<RefPointType> (*reference_, *result_cloud, transformation);

                {
                    pcl::visualization::PointCloudColorHandlerCustom<RefPointType> red_color (result_cloud, 0, 0, 255);
                    if (!viz.updatePointCloud (result_cloud, red_color, "resultcloud"))
                        viz.addPointCloud (result_cloud, red_color, "resultcloud");
                }

            }

        /*        void copy_pcl_clouds(CloudPtr& source, pcl::PointCloud<PointType_2>::Ptr& destination){

                ROS_WARN_STREAM("Source cloud size is : " << source->points.size());
                destination->points.resize(source->size());
                for(size_t i = 0; i < source->points.size(); i++){
                        destination->points[i].x = source->points[i].x;
                        destination->points[i].y = source->points[i].y;
                        destination->points[i].z = source->points[i].z;
                    }
            }*/

        boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (
                pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
            {
                // --------------------------------------------------------
                // -----Open 3D viewer and add point cloud and normals-----
                // --------------------------------------------------------
                boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
                viewer->setBackgroundColor (0, 0, 0);
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
                viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
                viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
                viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 0.05, "normals");

                viewer->addCoordinateSystem (1.0);
                viewer->initCameraParameters ();
                return (viewer);
            }

        void
        viz_cb (pcl::visualization::PCLVisualizer& viz)
            {
                //boost::mutex::scoped_lock lock (mtx_);
                std::chrono::time_point<std::chrono::high_resolution_clock> viz_init;
                if (print_time_)
                    viz_init = std::chrono::high_resolution_clock::now();

                std::chrono::time_point<std::chrono::high_resolution_clock> viz_setup;
                if (print_time_)
                    viz_setup = std::chrono::high_resolution_clock::now();
                PCL_INFO("VIZ_CB 00000000000 ......\n");
                viz.setBackgroundColor(0, 0, 0);
                viz.setCameraClipDistances(0.00884782, 8);
                viz.setCameraPosition( 0.0926632, 0.158074, -0.955283, 0.0926631, 0.158074, -0.955282, 0.0229289, -0.994791, -0.0993251);
                viz.setCameraFieldOfView(0.7);
                viz.setSize(960, 716);
                viz.setPosition(250, 52);

                if (!cloud_pass_ || !cloud_pass_downsampled_)
                    {
                        boost::this_thread::sleep (boost::posix_time::seconds (1));
                        return;
                    }

                PCL_INFO("VIZ_CB 1111111111111 ......\n");
                //ROS_WARN_STREAM("Downsampled cloud size : " << cloud_pass_downsampled_->points.size());
                //ROS_WARN_STREAM("New cloud size : " << new_cloud_->points.size());
                if (new_cloud_ && cloud_pass_downsampled_)
                    {
                        PCL_INFO("VIZ_CB 222222222222 ......\n");
                        //CloudPtr cloud_pass;
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pass;

                        //const pcl::PointCloud<pcl::PointXYZRGBNormal> test_2_ = *cloud_pass_;
                        //pcl::copyPointCloud(test_2_, test_1_);

                        if (!visualize_non_downsample_){
                                const pcl::PointCloud<pcl::PointXYZRGBNormal> test_2_ = *cloud_pass_downsampled_;
                                pcl::copyPointCloud(test_2_, test_1_);
                                //pcl::PointCloud<pcl::PointXYZRGB> test_3 = test_1_;
                                //ROS_INFO_STREAM("Size of test_1_ after pcl copycloud 11111111111 : " << test_3.size());
                                //cloud_pass = cloud_pass_downsampled_;
                                //pcl::copyPointCloud(*cloud_pass_downsampled_, *cloud_pass);
                                //copy_pcl_clouds(cloud_pass_downsampled_, cloud_pass);
                            }
                        else{
                                const pcl::PointCloud<pcl::PointXYZRGBNormal> test_2_ = *cloud_pass_;
                                pcl::copyPointCloud(test_2_, test_1_);
                                //pcl::PointCloud<pcl::PointXYZRGB> test_3 = test_1_;
                                //ROS_INFO_STREAM("Size of test_1_ after pcl copycloud 2222222222222 : " << test_3.size());
                                //cloud_pass = cloud_pass_;
                                //pcl::copyPointCloud(*cloud_pass_, *cloud_pass);
                                //copy_pcl_clouds(cloud_pass_, cloud_pass);
                            }


                        //cloud_pass.reset(&test_1_);
                        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pass(&test_1_);
                        cloud_pass = test_1_.makeShared();

                        //vis_cloud_.reset(&test_1_);

                        //convert cloud_pass from point type pcl::pclpointxyzrgbnormal to just pcl::pclxyzrgba !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        if (!viz.updatePointCloud (cloud_pass, "cloudpass"))
                        //if (!viz.updatePointCloud (vis_cloud_, "cloudpass"))
                            {
                                viz.addPointCloud (cloud_pass, "cloudpass");
                                //viz.addPointCloud (vis_cloud_, "cloudpass");
                                viz.resetCameraViewpoint ("cloudpass");
                            }
                        PCL_INFO("VIZ_CB 33333333333 ......\n");
                    }

                if (new_cloud_ && reference_)
                    {
                        bool ret = drawParticles (viz);
                        if (ret)
                            {
                                drawResult (viz);

                                // draw some texts
                                viz.removeShape ("N");
                                viz.addText ((boost::format ("number of Reference PointClouds: %d") % tracker_->getReferenceCloud ()->points.size ()).str (),
                                             10, 20, 20, 1.0, 1.0, 1.0, "N");

                                viz.removeShape ("M");
                                viz.addText ((boost::format ("number of Measured PointClouds:  %d") % cloud_pass_downsampled_->points.size ()).str (),
                                             10, 40, 20, 1.0, 1.0, 1.0, "M");

                                viz.removeShape ("tracking");
                                viz.addText ((boost::format ("tracking:        %f fps") % (1.0 / tracking_time_)).str (),
                                             10, 60, 20, 1.0, 1.0, 1.0, "tracking");

                                viz.removeShape ("downsampling");
                                viz.addText ((boost::format ("downsampling:    %f fps") % (1.0 / downsampling_time_)).str (),
                                             10, 80, 20, 1.0, 1.0, 1.0, "downsampling");

                                viz.removeShape ("computation");
                                viz.addText ((boost::format ("computation:     %f fps") % (1.0 / computation_time_)).str (),
                                             10, 100, 20, 1.0, 1.0, 1.0, "computation");

                                viz.removeShape ("particles");
                                viz.addText ((boost::format ("particles:     %d") % tracker_->getParticles ()->points.size ()).str (),
                                             10, 120, 20, 1.0, 1.0, 1.0, "particles");

                            }
                    }
                new_cloud_ = false;
            }

        void filterPassThrough (const CloudConstPtr &cloud, Cloud &result)
            {
                FPS_CALC_BEGIN;
                pcl::PassThrough<PointType> pass;
                pass.setFilterFieldName ("z");
                //pass.setFilterLimits (0.0, 10.0);
                pass.setFilterLimits (0.0, 1.5);
                //pass.setFilterLimits (0.0, 0.6);
                pass.setKeepOrganized (false);
                pass.setInputCloud (cloud);
                pass.filter (result);
                FPS_CALC_END("filterPassThrough");
            }

        void euclideanSegment (const CloudConstPtr &cloud,
                               std::vector<pcl::PointIndices> &cluster_indices)
            {
                FPS_CALC_BEGIN;
                pcl::EuclideanClusterExtraction<PointType> ec;
                KdTreePtr tree (new KdTree ());

                ec.setClusterTolerance (0.05); // 2cm
                ec.setMinClusterSize (50);
                ec.setMaxClusterSize (25000);
                //ec.setMaxClusterSize (400);
                ec.setSearchMethod (tree);
                ec.setInputCloud (cloud);
                ec.extract (cluster_indices);
                FPS_CALC_END("euclideanSegmentation");
            }

        void gridSample (const CloudConstPtr &cloud, Cloud &result, double leaf_size = 0.01)
            {
                FPS_CALC_BEGIN;
                double start = pcl::getTime ();
                pcl::VoxelGrid<PointType> grid;
                //pcl::ApproximateVoxelGrid<PointType> grid;
                grid.setLeafSize (float (leaf_size), float (leaf_size), float (leaf_size));
                grid.setInputCloud (cloud);
                grid.filter (result);
                //result = *cloud;
                double end = pcl::getTime ();
                downsampling_time_ = end - start;
                FPS_CALC_END("gridSample");
            }

        void gridSampleApprox (const CloudConstPtr &cloud, Cloud &result, double leaf_size = 0.01)
            {
                FPS_CALC_BEGIN;
                double start = pcl::getTime ();
                //pcl::VoxelGrid<PointType> grid;
                pcl::ApproximateVoxelGrid<PointType> grid;
                grid.setLeafSize (static_cast<float> (leaf_size), static_cast<float> (leaf_size), static_cast<float> (leaf_size));
                grid.setInputCloud (cloud);
                grid.filter (result);
                //result = *cloud;
                double end = pcl::getTime ();
                downsampling_time_ = end - start;
                FPS_CALC_END("gridSample");
            }

        void planeSegmentation (const CloudConstPtr &cloud,
                                pcl::ModelCoefficients &coefficients,
                                pcl::PointIndices &inliers)
            {
                FPS_CALC_BEGIN;
                pcl::SACSegmentation<PointType> seg;
                seg.setOptimizeCoefficients (true);
                seg.setModelType (pcl::SACMODEL_PLANE);
                seg.setMethodType (pcl::SAC_RANSAC);
                seg.setMaxIterations (1000);
                seg.setDistanceThreshold (0.03);
                seg.setInputCloud (cloud);
                seg.segment (inliers, coefficients);
                FPS_CALC_END("planeSegmentation");
            }

        void planeProjection (const CloudConstPtr &cloud,
                              Cloud &result,
                              const pcl::ModelCoefficients::ConstPtr &coefficients)
            {
                FPS_CALC_BEGIN;
                pcl::ProjectInliers<PointType> proj;
                proj.setModelType (pcl::SACMODEL_PLANE);
                proj.setInputCloud (cloud);
                proj.setModelCoefficients (coefficients);
                proj.filter (result);
                FPS_CALC_END("planeProjection");
            }

        void convexHull (const CloudConstPtr &cloud,
                         Cloud &,
                         std::vector<pcl::Vertices> &hull_vertices)
            {
                FPS_CALC_BEGIN;
                pcl::ConvexHull<PointType> chull;
                chull.setInputCloud (cloud);
                chull.reconstruct (*cloud_hull_, hull_vertices);
                FPS_CALC_END("convexHull");
            }

        void normalEstimation (const CloudConstPtr &cloud,
                               pcl::PointCloud<pcl::Normal> &result)
            {
                FPS_CALC_BEGIN;
                ne_.setInputCloud (cloud);
                ne_.compute (result);
                FPS_CALC_END("normalEstimation");
            }

        void tracking (const RefCloudConstPtr &cloud)
            {
                double start = pcl::getTime ();
                FPS_CALC_BEGIN;
                tracker_->setInputCloud (cloud);
                tracker_->compute ();
                double end = pcl::getTime ();
                FPS_CALC_END("tracking");
                tracking_time_ = end - start;
            }

        void addNormalToCloud (const CloudConstPtr &cloud,
                               const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                               RefCloud &result)
            {
                result.width = cloud->width;
                result.height = cloud->height;
                result.is_dense = cloud->is_dense;
                for (size_t i = 0; i < cloud->points.size (); i++)
                    {
                        RefPointType point;
                        point.x = cloud->points[i].x;
                        point.y = cloud->points[i].y;
                        point.z = cloud->points[i].z;
                        point.rgba = cloud->points[i].rgba;
                        point.normal[0] = normals->points[i].normal[0];
                        point.normal[1] = normals->points[i].normal[1];
                        point.normal[2] = normals->points[i].normal[2];
                        result.points.push_back (point);
                    }
            }

        void extractNonPlanePoints (const CloudConstPtr &cloud,
                                    const CloudConstPtr &cloud_hull,
                                    Cloud &result)
            {
                pcl::ExtractPolygonalPrismData<PointType> polygon_extract;
                pcl::PointIndices::Ptr inliers_polygon (new pcl::PointIndices ());
                polygon_extract.setHeightLimits (0.01, 10.0);
                polygon_extract.setInputPlanarHull (cloud_hull);
                polygon_extract.setInputCloud (cloud);
                polygon_extract.segment (*inliers_polygon);
                {
                    pcl::ExtractIndices<PointType> extract_positive;
                    extract_positive.setNegative (false);
                    extract_positive.setInputCloud (cloud);
                    extract_positive.setIndices (inliers_polygon);
                    extract_positive.filter (result);
                }
            }

        void removeZeroPoints (const CloudConstPtr &cloud,
                               Cloud &result)
            {
                for (size_t i = 0; i < cloud->points.size (); i++)
                    {
                        PointType point = cloud->points[i];
                        if (!(fabs(point.x) < 0.01 &&
                              fabs(point.y) < 0.01 &&
                              fabs(point.z) < 0.01) &&
                                !pcl_isnan(point.x) &&
                                !pcl_isnan(point.y) &&
                                !pcl_isnan(point.z))
                            result.points.push_back(point);
                    }

                result.width = static_cast<pcl::uint32_t> (result.points.size ());
                result.height = 1;
                result.is_dense = true;
            }

        void extractSegmentCluster (const CloudConstPtr &cloud,
                                    const std::vector<pcl::PointIndices> cluster_indices,
                                    const int segment_index,
                                    Cloud &result)
            {
                pcl::PointIndices segmented_indices = cluster_indices[segment_index];
                for (size_t i = 0; i < segmented_indices.indices.size (); i++)
                    {
                        PointType point = cloud->points[segmented_indices.indices[i]];
                        result.points.push_back (point);
                    }
                result.width = pcl::uint32_t (result.points.size ());
                result.height = 1;
                result.is_dense = true;
            }

        void
        cloud_cb (const sensor_msgs::PointCloud2ConstPtr &cloud_topic)
            {
                boost::mutex::scoped_lock lock (mtx_);
                double start = pcl::getTime ();
                FPS_CALC_BEGIN;
                //convert pointcloud2 topic into pcl::pointcloud
                pcl::PCLPointCloud2 pcl_pc2;
                pcl_conversions::toPCL(*cloud_topic, pcl_pc2);
                CloudPtr cloud (new Cloud);
                pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

                //the original code
                cloud_pass_.reset (new Cloud);
                cloud_pass_downsampled_.reset (new Cloud);
                pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
                pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
                filterPassThrough (cloud, *cloud_pass_);
                ROS_ERROR_STREAM("The count is : " << counter_);
                if (counter_ < 10)
                    {
                        gridSample (cloud_pass_, *cloud_pass_downsampled_, downsampling_grid_size_);
                    }
                else if (counter_ == 10)
                    {
                        //gridSample (cloud_pass_, *cloud_pass_downsampled_, 0.01);
                        cloud_pass_downsampled_ = cloud_pass_;
                        CloudPtr target_cloud;
                        if (use_convex_hull_)
                            {
                                planeSegmentation (cloud_pass_downsampled_, *coefficients, *inliers);
                                if (inliers->indices.size () > 3)
                                    {
                                        CloudPtr cloud_projected (new Cloud);
                                        cloud_hull_.reset (new Cloud);
                                        nonplane_cloud_.reset (new Cloud);

                                        planeProjection (cloud_pass_downsampled_, *cloud_projected, coefficients);
                                        convexHull (cloud_projected, *cloud_hull_, hull_vertices_);

                                        extractNonPlanePoints (cloud_pass_downsampled_, cloud_hull_, *nonplane_cloud_);
                                        target_cloud = nonplane_cloud_;
                                    }
                                else
                                    {
                                        PCL_WARN ("cannot segment plane\n");
                                    }
                            }
                        else
                            {
                                PCL_WARN ("without plane segmentation\n");
                                target_cloud = cloud_pass_downsampled_;
                            }

                        if (target_cloud != NULL)
                            {
                                PCL_INFO ("segmentation, please wait...\n");
                                std::vector<pcl::PointIndices> cluster_indices;
                                euclideanSegment (target_cloud, cluster_indices);
                                if (cluster_indices.size () > 0)
                                    {
                                        // select the cluster to track
                                        CloudPtr temp_cloud (new Cloud);
                                        extractSegmentCluster (target_cloud, cluster_indices, 0, *temp_cloud);
                                        Eigen::Vector4f c;
                                        pcl::compute3DCentroid<RefPointType> (*temp_cloud, c);
                                        int segment_index = 0;
                                        double segment_distance = c[0] * c[0] + c[1] * c[1];
                                        for (size_t i = 1; i < cluster_indices.size (); i++)
                                            {
                                                temp_cloud.reset (new Cloud);
                                                extractSegmentCluster (target_cloud, cluster_indices, int (i), *temp_cloud);
                                                pcl::compute3DCentroid<RefPointType> (*temp_cloud, c);
                                                double distance = c[0] * c[0] + c[1] * c[1];
                                                if (distance < segment_distance)
                                                    {
                                                        segment_index = int (i);
                                                        segment_distance = distance;
                                                    }
                                            }

                                        segmented_cloud_.reset (new Cloud);
                                        extractSegmentCluster (target_cloud, cluster_indices, segment_index, *segmented_cloud_);
                                        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
                                        normalEstimation (segmented_cloud_, *normals);
                                        RefCloudPtr ref_cloud (new RefCloud);
                                        addNormalToCloud (segmented_cloud_, normals, *ref_cloud);
                                        ref_cloud = segmented_cloud_;
                                        RefCloudPtr nonzero_ref (new RefCloud);
                                        removeZeroPoints (ref_cloud, *nonzero_ref);

                                        PCL_INFO ("calculating cog\n");

                                        RefCloudPtr transed_ref (new RefCloud);
                                        pcl::compute3DCentroid<RefPointType> (*nonzero_ref, c);
                                        Eigen::Affine3f trans = Eigen::Affine3f::Identity ();
                                        trans.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
                                        pcl::transformPointCloudWithNormals<RefPointType> (*ref_cloud, *transed_ref, trans.inverse());
                                        //pcl::transformPointCloud<RefPointType> (*nonzero_ref, *transed_ref, trans.inverse());
                                        CloudPtr transed_ref_downsampled (new Cloud);
                                        gridSample (transed_ref, *transed_ref_downsampled, downsampling_grid_size_);

                                        //lets see how the reference cloud looks like try to visualise it ...................................
                                        //convert pcl::pointcloud into pointcloud2 topic
                                        pcl::toROSMsg(*transed_ref_downsampled, *_reference_cloud_msgs);
                                        _reference_cloud_msgs->header.frame_id = "base";
                                        start_publishing_reference_cloud_ = true;


                                        //pcl::copyPointCloud()
                                        //const pcl::PointCloud<pcl::PointXYZRGBNormal> test_2_ = *segmented_cloud_;
                                        //PCL_INFO("I am here 44444444444 ......\n");
                                        //pcl::copyPointCloud(test_2_, test_1_);
                                        //ROS_INFO_STREAM("I am here 555555555555 ...... the size of cloud test_1 is : " << test_1_.size());
                                        //vis_cloud_.reset(&test_1_);
                                        //PCL_INFO("I am here 666666666666 ......\n");
                                        //viewer_ = normalsVis(vis_cloud_, normals);
                                        //PCL_INFO("I am here 7777777777 ......\n");

                                        tracker_->setReferenceCloud (transed_ref_downsampled);
                                        tracker_->setTrans (trans);
                                        reference_ = transed_ref;
                                        tracker_->setMinIndices (int (ref_cloud->points.size ()) / 2);
                                        //viz_cb(viewer_);
                                        PCL_INFO("I am here ************** ......\n");
                                    }
                                else
                                    {
                                        PCL_WARN ("euclidean segmentation failed\n");
                                    }
                            }
                    }
                else
                    {
                        //cloud_pass_downsampled_ = cloud_pass_;
                        //*cloud_pass_downsampled_ = *cloud_pass_;
                        //cloud_pass_downsampled_ = cloud_pass_;
                        gridSampleApprox (cloud_pass_, *cloud_pass_downsampled_, downsampling_grid_size_);
                        ROS_ERROR_STREAM("Downsampled Cloud Size : " << cloud_pass_downsampled_->size());
                        normals_.reset (new pcl::PointCloud<pcl::Normal>);
                        normalEstimation (cloud_pass_downsampled_, *normals_);
                        RefCloudPtr tracking_cloud (new RefCloud ());
                        addNormalToCloud (cloud_pass_downsampled_, normals_, *tracking_cloud);
                        tracking_cloud = cloud_pass_downsampled_;
                        tracking (cloud_pass_downsampled_);
                    }
                new_cloud_ = true;
                double end = pcl::getTime ();
                computation_time_ = end - start;
                FPS_CALC_END("computation");
                counter_++;
                viz_cb(viewer_);
            }

        void
        run (int argc, char** argv)
            {
                std::cout << "RUN" << std::endl;
                ros::init(argc, argv, "the_run_node");
                _cloud_sub = _nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/sw_registered/points", 1, &OpenNISegmentTracking::cloud_cb, this);
                _reference_cloud_pub = _nh.advertise<sensor_msgs::PointCloud2>("/reference_cloud", 1);
                _reference_cloud_msgs.reset(new sensor_msgs::PointCloud2);

                ros::Rate my_rate(20);
                //viewer_.registerPointPickingCallback();
                //viewer_.runOnVisualizationThread (boost::bind(&OpenNISegmentTracking::viz_cb, this, _1), "viz_cb");
                //viewer_.
                //                while(!start_publishing_reference_cloud_){
                //                        ros::spinOnce();
                //                        my_rate.sleep();
                //                    }

                //ros::spin();
                //while (!viewer_->wasStopped() && ros::ok()){
                while (ros::ok()){
                        viewer_.spinOnce(100, true);
                        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
                        ros::spinOnce();
                        /*if(start_publishing_reference_cloud_){
                                _reference_cloud_msgs->header.stamp = ros::Time::now();
                                _reference_cloud_pub.publish(*_reference_cloud_msgs);
                            }*/
                        my_rate.sleep();
                        //boost::this_thread::sleep(boost::posix_time::milliseconds(100));
                    }
            }

        //ros additional variables
        ros::NodeHandle _nh;
        ros::Subscriber _cloud_sub;
        ros::Publisher _reference_cloud_pub;
        pcl::PointCloud<pcl::PointXYZRGB> test_1_;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud_;
        bool print_time_;

        //original variables
        pcl::visualization::PCLVisualizer viewer_;
        //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
        pcl::PointCloud<pcl::Normal>::Ptr normals_;
        CloudPtr cloud_pass_;
        CloudPtr cloud_pass_downsampled_;
        CloudPtr plane_cloud_;
        CloudPtr nonplane_cloud_;
        CloudPtr cloud_hull_;
        CloudPtr segmented_cloud_;
        CloudPtr reference_;
        std::vector<pcl::Vertices> hull_vertices_;
        sensor_msgs::PointCloud2Ptr _reference_cloud_msgs;

        //std::string device_id_;
        boost::mutex mtx_;
        bool new_cloud_;
        pcl::NormalEstimationOMP<PointType, pcl::Normal> ne_; // to store threadpool
        boost::shared_ptr<ParticleFilter> tracker_;
        int counter_;
        bool use_convex_hull_;
        bool visualize_non_downsample_, start_publishing_reference_cloud_ = false;
        bool visualize_particles_;
        double tracking_time_;
        double computation_time_;
        double downsampling_time_;
        double downsampling_grid_size_;
    };

int
main (int argc, char** argv)
    {
        ros::init(argc, argv, "tracking_sample_node");

        bool use_convex_hull = true;
        bool visualize_non_downsample = false;
        bool visualize_particles = true;
        bool use_fixed = false;
        bool print_time = false;

        double downsampling_grid_size = 0.01;

        // the tracker
        //OpenNISegmentTracking<pcl::PointXYZRGBA> v (8, downsampling_grid_size,
        //                                          use_convex_hull,
        //                                        visualize_non_downsample, visualize_particles,
        //                                      use_fixed, print_time);
        OpenNISegmentTracking<pcl::PointXYZRGBNormal> v (8, downsampling_grid_size,
                                                         use_convex_hull,
                                                         visualize_non_downsample, visualize_particles,
                                                         use_fixed, print_time);
        v.run (argc, argv);
    }
