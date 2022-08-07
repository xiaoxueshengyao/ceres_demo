/***
 * 简答的BA问题
 * **/

#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// BAL数据集中的1次观测定义为一个相机位姿+一个空间点3D坐标+空间点2D像素坐标

class BALProblem
{
public:
    ~BALProblem()
    {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    int num_observations()          const{return num_observations_;}
    const double* observations()    const{return observations_;}
    double* mutable_cameras()       {return parameters_;}
    double* mutable_points()        {return parameters_ + 9*num_cameras_;}

    double* mutable_camera_for_observation(int i)
    {
        return mutable_cameras() + camera_index_[i]*9;
    }

    double* mutable_point_for_observation(int i)
    {
        return mutable_points() + point_index_[i] * 3;
    }

    bool LoadFile(const char* filename)
    {
        FILE* fptr = fopen(filename,"r");
        std::cout<<"read ok"<<std::endl;
        if(fptr == nullptr)
        {
            return false;
        };

        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_poinst_);
        FscanfOrDie(fptr, "%d", &num_observations_);

        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2* num_observations_];

        num_parameters_ = 9*num_cameras_ + 3*num_poinst_;
        parameters_ = new double[num_parameters_];


        for(int i=0; i<num_observations_; ++i)
        {
            FscanfOrDie(fptr, "%d", camera_index_+i);
            FscanfOrDie(fptr, "%d", point_index_ +i);
            for(int j=0; j<2; j++)
            {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i +j);
            }
        }

        for(int i=0; i<num_parameters_; i++)
        {
            FscanfOrDie(fptr, "%lf", parameters_+i);
        }
        std::cout<<"load ok"<<std::endl;
        return true;       

    }

private:

    template<typename T>
    void FscanfOrDie(FILE* fptr,const char* format,T *value)
    {
        int num_scanned = fscanf(fptr,format,value);
        if(num_scanned != 1){
            LOG(FATAL)<<"Invalid UW data file.";
        }
    }

    int num_cameras_;
    int num_poinst_;
    int num_observations_;
    int num_parameters_;

    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;

};



// The camera is parameterized using 9 parameters:
//  3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion.
struct SnavelyReprojectionError
{
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x),observed_y(observed_y){}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const 
    {
        // camera[0,1,2] are the angle-axis rotation.旋转
        // 下面相当于对点先做旋转在平移，用外参投影到像素坐标
        T p[3];
        ceres::AngleAxisRotatePoint(camera,point,p);

        //camera[3,4,5] are the translation.平移
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center of distortion
        // 投影到归一化坐标系
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = 1.0 + r2*(l1 + l2 * r2);

        //重投影误差
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        return true;

    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,9,3>(
                new SnavelyReprojectionError(observed_x,observed_y)));
    }


    double observed_x;
    double observed_y;
};



int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    if(argc != 2)
    {
        std::cerr <<"usage: ./ba <bal_problem>\n";
        return 1;
    }

    BALProblem bal_problem;
    if(!bal_problem.LoadFile(argv[1]))
    {
        std::cerr<<"file load error "<<argv[1]<<std::endl;
        return 1;
    }

    const double* observations = bal_problem.observations();

    ceres::Problem problem;
    for(int i=0; i<bal_problem.num_observations(); i++)
    {
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
            observations[2*i+0],observations[2*i+1]);
        problem.AddResidualBlock(cost_function,
                                NULL,
                                bal_problem.mutable_camera_for_observation(i),
                                bal_problem.mutable_point_for_observation(i));
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options opt;
    opt.linear_solver_type = ceres::DENSE_SCHUR;
    opt.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary sum;
    ceres::Solve(opt,&problem,&sum);
    std::cout<<sum.BriefReport()<<"\n";

    return 0;

}