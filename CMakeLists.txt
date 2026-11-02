cmake_minimum_required(VERSION 3.8)
project(curtmini_piper_mujoco_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/build_model scripts/scan_relay
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_robot_model test/test_robot_model.py TIMEOUT 300)
  ament_add_pytest_test(test_worlds test/test_worlds.py TIMEOUT 300)
endif()

ament_package()
