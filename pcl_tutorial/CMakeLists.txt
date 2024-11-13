cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pcl_turtorial)

SET(CMAKE_INSTALL_PREFIX ./)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#过滤z轴0-1之外的所有点
add_executable (passthrough passthrough.cpp)
target_link_libraries (passthrough ${PCL_LIBRARIES})

#体素化降采样
add_executable (voxel_grid voxel_grid.cpp)
target_link_libraries (voxel_grid ${PCL_LIBRARIES})

#标准差过滤异常值
add_executable (statistical_removal statistical_removal.cpp)
target_link_libraries (statistical_removal ${PCL_LIBRARIES})

#投影滤波到平面
add_executable (project_inliers project_inliers.cpp)
target_link_libraries (project_inliers ${PCL_LIBRARIES})

#从点云中获取平面特征
add_executable (extract_indices extract_indices.cpp)
target_link_libraries (extract_indices ${PCL_LIBRARIES})

#球面半径内剔除点云异常值，或者设置条件例如筛掉z轴某个范围内
add_executable (remove_outliers remove_outliers.cpp)
target_link_libraries (remove_outliers ${PCL_LIBRARIES})

#求解法向量例程
add_executable (normal_test normal.cpp)
target_link_libraries (normal_test ${PCL_LIBRARIES})

#点云旋转矩阵变换
add_executable (matrix_transform matrix_transform.cpp)
target_link_libraries (matrix_transform ${PCL_LIBRARIES})

#使用积分图求解点云法向量
add_executable (normal_integral_image normal_estimation_using_integral_images.cpp)
target_link_libraries (normal_integral_image ${PCL_LIBRARIES})

#kdTree求解邻域点
add_executable (kdtree_search kdtree_search.cpp)
target_link_libraries (kdtree_search ${PCL_LIBRARIES})

#octree八叉树求解邻域点
add_executable (octree_search octree_search.cpp)
target_link_libraries (octree_search ${PCL_LIBRARIES})

#基于多项式的点云重采样平滑方法
add_executable (resampling resampling.cpp)
target_link_libraries (resampling ${PCL_LIBRARIES})

#投影滤波到平面，并计算凹凸包
add_executable (concave_hull_2d concave_hull_2d.cpp)
target_link_libraries (concave_hull_2d ${PCL_LIBRARIES})

install(TARGETS passthrough voxel_grid statistical_removal project_inliers 
    extract_indices remove_outliers normal_test matrix_transform 
    normal_integral_image kdtree_search octree_search resampling concave_hull_2d
        DESTINATION ../)