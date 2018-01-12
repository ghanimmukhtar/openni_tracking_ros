cmake_minimum_required(VERSION 2.8.3)
project(openni_tracking_ros)

find_package(PCL 1.7 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fPIC")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_ros
  roscpp
  std_msgs
)




#find_package(Eigen3 REQUIRED)

find_package(OpenCV 2.4.8 REQUIRED COMPONENTS features2d nonfree)
include_directories(${OpenCV_INCLUDE_DIRS})


#link_directories(${PCL_LIBRARY_DIRS} )
#add_definitions(${PCL_DEFINITIONS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES openni_tracking_ros
#  CATKIN_DEPENDS cv_bridge pcl_ros roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable (testyy src/testyy.cpp)
target_link_libraries (testyy ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (tracking_sample src/tracking_sample.cpp)
target_link_libraries (tracking_sample ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (planar_segmentation src/planar_segmentation.cpp)
target_link_libraries (planar_segmentation ${catkin_LIBRARIES} ${PCL_LIBRARIES})