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

char filename[] = "../model/Adroit/Adroit_hand.xml";

// for trajectory
double a0[12]={0},a1[12]={0},a2[12]={0},a3[12]={0};
double qref[12]={0}, uref[12]={0};
double g0[12]={0},g1[12]={0};
//for udp
int flag;
int init = 0;
int state_curr;
double t_current;
double q_curr[12] = {0};
// for finite state machine
double state = 0;
#define fist_0 0
#define fist_1 1
#define fist_2 2
#define fist_3 3

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

void general_trajectory(){
    double q_0[] = {0,0,0,0,0,0,0,0,0,0,0,0};

    double q_f[] = {1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6};
    double di = 0.16;
    double df = 0.05;
    for(int i=0;i<12;i++){
        g0[i] = (q_0[i]*df-q_f[i]*di)/(df-di); 
        g1[i] = (q_f[i]-q_0[i])/(df-di);
    }


    
}

void generate_trajectory(double t0, double tf, double q_0[12],double q_f[12])
{
  int i;
  double tf_t0_3 = (tf-t0)*(tf-t0)*(tf-t0);
  for (i=0;i<12;i++)
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


void init_controller(const mjModel* m, mjData* d)
{ 
  
  
  general_trajectory();
  double kp = 5;
  state_curr = fist_2;
  state = fist_2;
  set_Gains(m,3,kp);
  set_Gains(m,4,kp);
  set_Gains(m,5,kp);
  set_Gains(m,7,kp);
  set_Gains(m,8,kp);
  set_Gains(m,9,kp);
  set_Gains(m,11,kp);
  set_Gains(m,12,kp);
  set_Gains(m,13,kp);
  set_Gains(m,16,kp);
  set_Gains(m,17,kp);
  set_Gains(m,18,kp);
}





//customized controller
void controller(const mjModel* m, mjData* d)
{ 
    int i;
    double t;
    t = d->time;
    
    
    //int state_ext;
    

  //printf("%f",t);
  //printf("\n");
  double pos_init[] = {0,0,0,0,0,0,0,0,0,0,0,0};
  double pos_0[] = {0.53,0.37,0.3,0.53,0.37,0.3,0.53,0.37,0.3,0.53,0.37,0.3};
  double pos_1[] = {1.23,0.96,0.75,1.23,0.96,0.75,1.23,0.96,0.75,1.23,0.96,0.75};
  double pos_2[] = {1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6};
  
  //printf("%d",flag);
  if (flag == 1){
      //printf("here");
      flag = 0;
  for (i=0;i<12;i++)
    {
      uref[i] = g0[i] + g1[i]*state;
      printf("%f",uref[i]);
      //printf("%f ",qref[i]);

    }
   printf("\n");
   q_curr[0]= d->qpos[3]; q_curr[1] = d->qpos[4]; q_curr[2]= d->qpos[5]; q_curr[3] = d->qpos[7]; q_curr[4]= d->qpos[8]; q_curr[5] = d->qpos[9];
   q_curr[6]= d->qpos[11]; q_curr[7] = d->qpos[12]; q_curr[8]= d->qpos[13]; q_curr[9] = d->qpos[16]; q_curr[10]= d->qpos[17]; q_curr[11] = d->qpos[18];
   t_current = d->time;
   printf("%f",state);
  }
  /*here 


  if (state_curr != state)
  {   //printf("here");
      //double q_curr[12] = {0};
      q_curr[0]= d->qpos[3]; q_curr[1] = d->qpos[4]; q_curr[2]= d->qpos[5]; q_curr[3] = d->qpos[7]; q_curr[4]= d->qpos[8]; q_curr[5] = d->qpos[9];
      q_curr[6]= d->qpos[11]; q_curr[7] = d->qpos[12]; q_curr[8]= d->qpos[13]; q_curr[9] = d->qpos[16]; q_curr[10]= d->qpos[17]; q_curr[11] = d->qpos[18];
      
      t_current = d->time;
      
      //state_ext = state;
      state_curr = state;
      //init = 1;
      
  }
  
  //printf("%d",state_curr);

  //printf("init:%d \n",state_curr);
  if (state_curr == fist_1 )
  {
    
    
    generate_trajectory(t_current,t_current+0.2,q_curr,pos_0);
    
  }

  if (state_curr == fist_2 )
  {
     generate_trajectory(t_current,t_current+0.2,q_curr,pos_1);
  }

  if (state_curr == fist_3 )
  {
    
    generate_trajectory(t_current,t_current+0.2,q_curr,pos_2);
  }
  
  
  if (state_curr == fist_0 )
  {
    
    generate_trajectory(t_current,t_current+0.2,q_curr,pos_init);
  }
  

  //printf("diff:%f \n",t-t_current);
  

  here*/
  
  generate_trajectory(t_current,t_current+0.1,q_curr,uref);


  //if (t<= t_current+0.1)
    for (i=0;i<12;i++)
    {
      qref[i] = a0[i] + a1[i]*t + a2[i]*t*t + a3[i]*t*t*t;

      //printf("%f ",qref[i]);

    }
    //printf("checkpoint:%f \n",t_current);
    d->ctrl[3] = qref[0];
    d->ctrl[4] = qref[1];
    d->ctrl[5] = qref[2];

    d->ctrl[7] = qref[3];
    d->ctrl[8] = qref[4];
    d->ctrl[9] = qref[5];

    d->ctrl[11] = qref[6];
    d->ctrl[12] = qref[7];
    d->ctrl[13] = qref[8];

    d->ctrl[16] = qref[9];
    d->ctrl[17] = qref[10];
    d->ctrl[18] = qref[11];

    
  


  /*
   if (state == fist_0)
  {
  d->ctrl[3] = pos_init[0];
  d->ctrl[4] = pos_init[1];
  d->ctrl[5] = pos_init[2];

  d->ctrl[7] = pos_init[0];
  d->ctrl[8] = pos_init[1];
  d->ctrl[9] = pos_init[2];

  d->ctrl[11] = pos_init[0];
  d->ctrl[12] = pos_init[1];
  d->ctrl[13] = pos_init[2];

  d->ctrl[16] = pos_init[0];
  d->ctrl[17] = pos_init[1];
  d->ctrl[18] = pos_init[2];
  state = fist_1;
  printf("0");
  }



   if (state == fist_1)
  {
  d->ctrl[3] = pos_0[0];
  d->ctrl[4] = pos_0[1];
  d->ctrl[5] = pos_0[2];

  d->ctrl[7] = pos_0[0];
  d->ctrl[8] = pos_0[1];
  d->ctrl[9] = pos_0[2];

  d->ctrl[11] = pos_0[0];
  d->ctrl[12] = pos_0[1];
  d->ctrl[13] = pos_0[2];

  d->ctrl[16] = pos_0[0];
  d->ctrl[17] = pos_0[1];
  d->ctrl[18] = pos_0[2];
  state = fist_2;
  printf("1");
  }


   if (state == fist_2)
  {
  d->ctrl[3] = pos_1[0];
  d->ctrl[4] = pos_1[1];
  d->ctrl[5] = pos_1[2];

  d->ctrl[7] = pos_1[0];
  d->ctrl[8] = pos_1[1];
  d->ctrl[9] = pos_1[2];

  d->ctrl[11] = pos_1[0];
  d->ctrl[12] = pos_1[1];
  d->ctrl[13] = pos_1[2];

  d->ctrl[16] = pos_1[0];
  d->ctrl[17] = pos_1[1];
  d->ctrl[18] = pos_1[2];
  state = fist_3;
  printf("2");
  }

  
   if (state == fist_3)
  {
  d->ctrl[3] = pos_2[0];
  d->ctrl[4] = pos_2[1];
  d->ctrl[5] = pos_2[2];

  d->ctrl[7] = pos_2[0];
  d->ctrl[8] = pos_2[1];
  d->ctrl[9] = pos_2[2];

  d->ctrl[11] = pos_2[0];
  d->ctrl[12] = pos_2[1];
  d->ctrl[13] = pos_2[2];

  d->ctrl[16] = pos_2[0];
  d->ctrl[17] = pos_2[1];
  d->ctrl[18] = pos_2[2];
  state = fist_0;
  printf("3");
  }
  
  */
  

  
     
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
                  
            double state_h = std::stod(data);
            state = state_h;
            flag = 1;
            //printf("%d",state);
        }
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
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
    double arr_view[] = {90,-45, 2, 0.000000, 0.000000, 0.000000};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];


    
    mjcb_control = controller;
    init_controller(m,d);

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