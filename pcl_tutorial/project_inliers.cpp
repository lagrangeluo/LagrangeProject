#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h> // for PointCloud
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

int main ()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);

//   // Fill in the cloud data
//   cloud->width  = 5;
//   cloud->height = 1;
//   cloud->points.resize (cloud->width * cloud->height);

//   for (auto& point: *cloud)
//   {
//     point.x = 1024 * rand () / (RAND_MAX + 1.0f);
//     point.y = 1024 * rand () / (RAND_MAX + 1.0f);
//     point.z = 1024 * rand () / (RAND_MAX + 1.0f);
//   }


//   std::cerr << "Cloud before projection: " << std::endl;
//   for (const auto& point: *cloud)
//     std::cerr << "    " << point.x << " "
//                         << point.y << " "
//                         << point.z << std::endl;

  // Fill in the cloud data
  pcl::PCDReader reader;
  // Replace the path below with the path where you saved your file
  reader.read<pcl::PointXYZ> ("table_scene_lms400.pcd", *cloud);

  std::cerr << "Cloud before filtering: " << std::endl;
  std::cerr << *cloud << std::endl;

  // Create a set of planar coefficients with X=Y=0,Z=1
  // 创建滤波投影系数对象
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  coefficients->values.resize (4);
  coefficients->values[0] = coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

  // Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZ> proj;  //创建投影滤波对象
  proj.setModelType (pcl::SACMODEL_PLANE);  //设置模型为平面
  proj.setInputCloud (cloud);
  proj.setModelCoefficients (coefficients); //传入系数
  proj.filter (*cloud_projected);

  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("table_scene_lms400_project_z=0.pcd", *cloud_projected, false);

//   std::cerr << "Cloud after projection: " << std::endl;
//   for (const auto& point: *cloud_projected)
//     std::cerr << "    " << point.x << " "
//                         << point.y << " "
//                         << point.z << std::endl;

  return (0);
}