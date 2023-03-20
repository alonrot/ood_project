cmake_minimum_required(VERSION 2.8.3)
project(predictions_module)

# include_directories(include) # amarco: commented out
# include_directories(include python_interface_go1/include /usr/local/include /usr/local/include/lcm /usr/local/include/eigen3) # amarco added this line
# include_directories(include /usr/local/include /usr/local/include/eigen3) # amarco added this line
include_directories(include include/eigen) # amarco added this line

# link_directories(lib /usr/local/lib)

add_compile_options(-std=c++11) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

# # amarco
# add_subdirectory(python_interface_go1/pybind11)
# pybind11_add_module(real_robot_interface_go1 python_interface_go1/src/real_robot_interface_go1.cpp)
# target_link_libraries(real_robot_interface_go1 ${EXTRA_LIBS})

# # amarco
# add_subdirectory(python_interface_go1/pybind11)
# pybind11_add_module(real_robot_interface_go1_highlevel python_interface_go1/src/real_robot_interface_go1_highlevel.cpp)
# target_link_libraries(real_robot_interface_go1_highlevel ${EXTRA_LIBS})

# # one pc one process
# add_executable(example_position examples/example_position.cpp)
# target_link_libraries(example_position ${EXTRA_LIBS})

# add_executable(stand_up_and_down examples/stand_up_and_down.cpp)
# target_link_libraries(stand_up_and_down ${EXTRA_LIBS})

# add_executable(example_velocity examples/example_velocity.cpp)
# target_link_libraries(example_velocity ${EXTRA_LIBS})

# add_executable(example_torque examples/example_torque.cpp)
# target_link_libraries(example_torque ${EXTRA_LIBS})

# add_executable(example_walk examples/example_walk.cpp)
# target_link_libraries(example_walk ${EXTRA_LIBS})

# # add_executable(example_wirelessHandle examples/example_wirelessHandle.cpp)
# # target_link_libraries(example_wirelessHandle ${EXTRA_LIBS})

# # multi pc
# add_executable(udp_send_test examples/multi_pc_udp_send.cpp)
# target_link_libraries(udp_send_test ${EXTRA_LIBS})

# add_executable(udp_recv_test examples/multi_pc_udp_recv.cpp)
# target_link_libraries(udp_recv_test ${EXTRA_LIBS})

# # lcm server
# add_executable(lcm_server examples/lcm_server.cpp)
# target_link_libraries(lcm_server ${EXTRA_LIBS})



# # amarco
# add_executable(latency_test_client examples/latency_test_client.cpp)
# target_link_libraries(latency_test_client ${EXTRA_LIBS})

# add_executable(latency_test_server examples/latency_test_server.cpp)
# target_link_libraries(latency_test_server ${EXTRA_LIBS})


# set(EXTRA_LIBS -pthread libunitree_legged_sdk_${ARCH}.so lcm) # amarco edited
set(EXTRA_LIBS -pthread) # amarco edited


add_executable(test_predictions tests/test_predictions.cpp src/predictions.cpp)
target_link_libraries(test_predictions ${EXTRA_LIBS})


# amarco
add_subdirectory(include/pybind11)
pybind11_add_module(predictions_interface src/predictions_interface.cpp src/predictions.cpp)
target_link_libraries(predictions_interface ${EXTRA_LIBS})




