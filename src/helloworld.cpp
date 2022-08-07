// 一个ceres简单的例子
// 最小化 0.5 * （10 - x）^2 使用自动求导进行雅克比计算


#include "ceres/ceres.h"
#include "glog/logging.h"
#include <iostream>


// 简单的残差计算
// r = 10 - x，operator()重载用于自动求导
struct CostFunctor
{
    //函数名后面的const不能省，函数体内不能对成员数据做任何改动
    template <typename T> bool operator()(const T* const x, T* residual) const
    {
        residual[0] = 10.0 - x[0];//要用10.0，否则报错
        return true;
    }
};


//数值求导方式的残差计算
struct NumericDiffCostFunctor
{
    bool operator()(const double* const x, double* residual) const
    {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

// <1,1> 分别是残差维度、参数块的维度
class QuadraticCostFunction : public ceres::SizedCostFunction<1,1>
{
    public:
        virtual ~QuadraticCostFunction() {}
        virtual bool Evaluate(double const* const* parameters,
                              double* residual,
                              double** jacobians) const
        {
            const double x = parameters[0][0];
            residual[0] = 10.0 - x;

            // 对于 f(x) = 10 - x 求导得  -1
            if(jacobians != nullptr && jacobians[0] != nullptr)
            {
                jacobians[0][0] = -1;
            }

            return true;
        
        }

};



int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    double x = 0.5;
    const double init_x = x;

    ceres::Problem problem;

    // 残差计算,三个参数分别是代价函数、残差维度、参数块维度（参数快维度可以设多个）
    // ceres::CostFunction* cost_function = 
    //     new ceres::AutoDiffCostFunction<CostFunctor,1,1>(new CostFunctor);

    
    //数值计算方式
    // ceres::CostFunction* cost_function = 
    //     new ceres::NumericDiffCostFunction<NumericDiffCostFunctor,ceres::CENTRAL,1,1>(
    //         new NumericDiffCostFunctor);

    // 解析法计算残差
    ceres::CostFunction* cost_function = new QuadraticCostFunction;
    
    //代价函数对象，核函数、待优化变量
    problem.AddResidualBlock(cost_function,nullptr,&x);


    //构建求解器
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary sum;
    ceres::Solve(options, &problem, &sum);

    std::cout<< sum.BriefReport() << std::endl;
    std::cout<<"x : "<<init_x<<" ----> "<<"x : "<< x <<std::endl;


    return 0;
}

