cmake_minimum_required(VERSION 3.5)
project(arena_evaluation_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BenchmarkState.msg"
  "srv/ChangeDirectory.srv"
  DEPENDENCIES builtin_interfaces
)

ament_package()
