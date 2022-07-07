//Franka Panda Mujoco Simulation
//Author: Liyuan Qi

#include <iostream>
#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <thread>
#include <chrono>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <json.hpp>

char filename[] = "../model/franka_sim/franka_panda.xml";

//for trajectory
double a0[7]={0},a1[7]={0},a2[7]={0},a3[7]={0};
double qref[7]={0}, uref[7]={0};
double g0[7]={0},g1[7]={0};
//for udp
int flag;
int init = 0;
int state_curr;
double t_current;
double q_curr[7] = {0};
// for finite state machine
double state[7];
// for json
using namespace nlohmann;





mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

//set gains
void set_Gains(const mjModel* m, int actuator_no, double kp)
{   
    m->actuator_gainprm[10*actuator_no+0] = kp; 
    m->actuator_biasprm[10*actuator_no+1] = -kp; 
}

void init_controller(const mjModel* m, mjData* d)
{ 
  
    double arr_pos[] = {0, -0.785, 0, -2.356, 0, 1.571, 0.785};

    d->ctrl[0] = /*d->qpos[0]+ 10*(d->qpos[0] -*/ arr_pos[0];
    d->ctrl[1] = /*d->qpos[1]+ 10*(d->qpos[1] -*/ arr_pos[1];
    d->ctrl[2] = /*d->qpos[2]+ 10*(d->qpos[2] -*/ arr_pos[2];
    d->ctrl[3] = /*d->qpos[3]+ 10*(d->qpos[3] -*/ arr_pos[3];
    d->ctrl[4] = /*d->qpos[4]+ 10*(d->qpos[4] -*/ arr_pos[4];
    d->ctrl[5] = /*d->qpos[5]+ 10*(d->qpos[5] -*/ arr_pos[5];
    d->ctrl[6] = /*d->qpos[6]+ 10*(d->qpos[6] -*/ arr_pos[6];

}

// cube trajectory generate
void generate_trajectory(double t0, double tf, double q_0[7],double q_f[7])
{
  int i;
  double tf_t0_3 = (tf-t0)*(tf-t0)*(tf-t0);
  for (i=0;i<7;i++)
  {
    double q0 = q_0[i], qf = q_f[i];
    a0[i] = qf*t0*t0*(3*tf-t0) + q0*tf*tf*(tf-3*t0); a0[i] = a0[i]/tf_t0_3;
    a1[i] = 6*t0*tf*(q0-qf); a1[i] = a1[i]/tf_t0_3;
    a2[i] = 3*(t0+tf)*(qf-q0); a2[i] = a2[i]/tf_t0_3;
    a3[i] = 2*(q0-qf); a3[i] = a3[i]/tf_t0_3;
  }
}

int create_and_bind(int port, int& len){

    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock_fd < 0)
    {
        perror("socket");
        exit(1);
    }


    struct sockaddr_in addr_serv;
    memset(&addr_serv, 0, sizeof(struct sockaddr_in));
    addr_serv.sin_family = AF_INET;
    addr_serv.sin_port = htons(port); 
    addr_serv.sin_addr.s_addr = htonl(INADDR_ANY); 
    len = sizeof(addr_serv);

    bind(sock_fd, (struct sockaddr *)&addr_serv, sizeof(addr_serv));

    return sock_fd; 
}

void thread_upd_recieve(){
    
    int len;

    int sock_fd = create_and_bind(2022, len);

    ssize_t recv_num;
    char recv_buf[200];
    struct sockaddr_in addr_client;
                    



    while (true)
    {   
        //start server
        recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *)&addr_client, (socklen_t *)&len);

        if(recv_num > 0){
            
            recv_buf[recv_num] = '\0';
            std::string data = recv_buf;
            std::cout << data << std::endl;
            //parse data here state_h -> double[7]
            //double state_h = std::stod(data);
            double state_h[7] = {0};
            auto j = json::parse(data);
            auto joints = j["joints"];
            // double _if_grasp = j["gripper"];

            double joint0 = joints[0];
            double joint1 = joints[1];
            double joint2 = joints[2];
            double joint3 = joints[3];
            double joint4 = joints[4];
            double joint5 = joints[5];
            double joint6 = joints[6];
            state_h[0] = joint0;
            state_h[1] = joint1;
            state_h[2] = joint2;
            state_h[3] = joint3;
            state_h[4] = joint4;
            state_h[5] = joint5;
            state_h[6] = joint6;
            
            for (int i=0;i<7;i++){
                state[i] = state_h[i];
                printf("%f",state[i]);
            }
            
            flag = 1;
            
        }
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}


//customized controller
void controller(const mjModel* m, mjData* d)
{   
    
    // int i;
    // double t;

    
    // double arr_pos[] = {0, -0.785, 0, -2.356, 0, 1.571, 0.785};
    // //double arr_pos[] = {0, -0.785, 0, -2.356, 0, 0, 0};
    

    

    // d->ctrl[0] = /*d->qpos[0]+ 10*(d->qpos[0] -*/ arr_pos[0];
    // d->ctrl[1] = /*d->qpos[1]+ 10*(d->qpos[1] -*/ arr_pos[1];
    // d->ctrl[2] = /*d->qpos[2]+ 10*(d->qpos[2] -*/ arr_pos[2];
    // d->ctrl[3] = /*d->qpos[3]+ 10*(d->qpos[3] -*/ arr_pos[3];
    // d->ctrl[4] = /*d->qpos[4]+ 10*(d->qpos[4] -*/ arr_pos[4];
    // d->ctrl[5] = /*d->qpos[5]+ 10*(d->qpos[5] -*/ arr_pos[5];
    // d->ctrl[6] = /*d->qpos[6]+ 10*(d->qpos[6] -*/ arr_pos[6];

    /*  get UDP data   */
    if (flag == 1){
        flag = 0;
    d->ctrl[0] = state[0];
    d->ctrl[1] = state[1];
    d->ctrl[2] = state[2];
    d->ctrl[3] = state[3];
    d->ctrl[4] = state[4];
    d->ctrl[5] = state[5];
    d->ctrl[6] = state[6];
    
    
    }    
//     for(i=0;i<7;i++){

//         printf("%f ",state[i]);
//     }
 }

// main function
int sim_main()
{   
    mj_activate("mjkey.txt");
    // load and compile model
    char error[1000] = "Could not load binary model";
    m = mj_loadXML(filename, 0, error, 1000);
    // check command-line arguments
    // if( argc<2 )
    //     m = mj_loadXML(filename, 0, error, 1000);

    // else
    //     if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
    //         m = mj_loadModel(argv[1], 0);
    //     else
    //         m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    //std::cout << m->nq <<std::endl;

    // make data
    //
    d = mj_makeData(m);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    //set the initial position of camera
    double arr_view[] = {90,-45, 4, 0.000000, 0.000000, 0.000000};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];


    init_controller(m,d);
    mjcb_control = controller;


    // run main loop, target real-time simulation and 60 fps rendering
    while( !glfwWindowShouldClose(window) )
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 ){
            mj_step(m, d);
        }
        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        //printf("{%f, %f, %f, %f, %f, %f};\n",cam.azimuth,cam.elevation, cam.distance,cam.lookat[0],cam.lookat[1],cam.lookat[2]);
        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}

int main(int argc, const char** argv){
    std::thread t1(sim_main);
    std::thread t2(thread_upd_recieve);
    
    t1.join();
    t2.join();
    
    return 1;
}