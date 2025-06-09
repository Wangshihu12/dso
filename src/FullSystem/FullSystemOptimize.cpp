/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ResidualProjections.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso
{





void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		PointFrameResidual* r = activeResiduals[k];
		(*stats)[0] += r->linearize(&Hcalib);

		if(fixLinearization)
		{
			r->applyRes(true);

			if(r->efResidual->isActive())
			{
				if(r->isNew)
				{
					PointHessian* p = r->point;
					Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u,p->v, 1);	// projected point assuming infinite depth.
					Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll*p->idepth_scaled;	// projected point with real depth.
					float relBS = 0.01*((ptp_inf.head<2>() / ptp_inf[2])-(ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.


					if(relBS > p->maxRelBaseline)
						p->maxRelBaseline = relBS;

					p->numGoodResiduals++;
				}
			}
			else
			{
				toRemove[tid].push_back(activeResiduals[k]);
			}
		}
	}
}


void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
		activeResiduals[k]->applyRes(true);
}
void FullSystem::setNewFrameEnergyTH()
{

	// collect all residuals and make decision on TH.
	allResVec.clear();
	allResVec.reserve(activeResiduals.size()*2);
	FrameHessian* newFrame = frameHessians.back();

	for(PointFrameResidual* r : activeResiduals)
		if(r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
		{
			allResVec.push_back(r->state_NewEnergyWithOutlier);

		}

	if(allResVec.size()==0)
	{
		newFrame->frameEnergyTH = 12*12*patternNum;
		return;		// should never happen, but lets make sure.
	}


	int nthIdx = setting_frameEnergyTHN*allResVec.size();

	assert(nthIdx < (int)allResVec.size());
	assert(setting_frameEnergyTHN < 1);

	std::nth_element(allResVec.begin(), allResVec.begin()+nthIdx, allResVec.end());
	float nthElement = sqrtf(allResVec[nthIdx]);






    newFrame->frameEnergyTH = nthElement*setting_frameEnergyTHFacMedian;
	newFrame->frameEnergyTH = 26.0f*setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH*(1-setting_frameEnergyTHConstWeight);
	newFrame->frameEnergyTH = newFrame->frameEnergyTH*newFrame->frameEnergyTH;
	newFrame->frameEnergyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;



//
//	int good=0,bad=0;
//	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
//	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
//			meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
//			good, bad);
}
Vec3 FullSystem::linearizeAll(bool fixLinearization)
{
	double lastEnergyP = 0;
	double lastEnergyR = 0;
	double num = 0;


	std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
	for(int i=0;i<NUM_THREADS;i++) toRemove[i].clear();

	if(multiThreading)
	{
		treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
		lastEnergyP = treadReduce.stats[0];
	}
	else
	{
		Vec10 stats;
		linearizeAll_Reductor(fixLinearization, toRemove, 0,activeResiduals.size(),&stats,0);
		lastEnergyP = stats[0];
	}


	setNewFrameEnergyTH();


	if(fixLinearization)
	{

		for(PointFrameResidual* r : activeResiduals)
		{
			PointHessian* ph = r->point;
			if(ph->lastResiduals[0].first == r)
				ph->lastResiduals[0].second = r->state_state;
			else if(ph->lastResiduals[1].first == r)
				ph->lastResiduals[1].second = r->state_state;



		}

		int nResRemoved=0;
		for(int i=0;i<NUM_THREADS;i++)
		{
			for(PointFrameResidual* r : toRemove[i])
			{
				PointHessian* ph = r->point;

				if(ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].first=0;
				else if(ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].first=0;

				for(unsigned int k=0; k<ph->residuals.size();k++)
					if(ph->residuals[k] == r)
					{
						ef->dropResidual(r->efResidual);
						deleteOut<PointFrameResidual>(ph->residuals,k);
						nResRemoved++;
						break;
					}
			}
		}
		//printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved, (int)activeResiduals.size());

	}

	return Vec3(lastEnergyP, lastEnergyR, num);
}




// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
//	float meanStepC=0,meanStepP=0,meanStepD=0;
//	meanStepC += Hcalib.step.norm();

	Vec10 pstepfac;
	pstepfac.segment<3>(0).setConstant(stepfacT);
	pstepfac.segment<3>(3).setConstant(stepfacR);
	pstepfac.segment<4>(6).setConstant(stepfacA);


	float sumA=0, sumB=0, sumT=0, sumR=0, sumID=0, numID=0;

	float sumNID=0;

	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
		for(FrameHessian* fh : frameHessians)
		{
			Vec10 step = fh->step;
			step.head<6>() += 0.5f*(fh->step_backup.head<6>());

			fh->setState(fh->state_backup + step);
			sumA += step[6]*step[6];
			sumB += step[7]*step[7];
			sumT += step.segment<3>(0).squaredNorm();
			sumR += step.segment<3>(3).squaredNorm();

			for(PointHessian* ph : fh->pointHessians)
			{
				float step = ph->step+0.5f*(ph->step_backup);
				ph->setIdepth(ph->idepth_backup + step);
				sumID += step*step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

                ph->setIdepthZero(ph->idepth_backup + step);
			}
		}
	}
	else
	{
		Hcalib.setValue(Hcalib.value_backup + stepfacC*Hcalib.step);
		for(FrameHessian* fh : frameHessians)
		{
			fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
			sumA += fh->step[6]*fh->step[6];
			sumB += fh->step[7]*fh->step[7];
			sumT += fh->step.segment<3>(0).squaredNorm();
			sumR += fh->step.segment<3>(3).squaredNorm();

			for(PointHessian* ph : fh->pointHessians)
			{
				ph->setIdepth(ph->idepth_backup + stepfacD*ph->step);
				sumID += ph->step*ph->step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

                ph->setIdepthZero(ph->idepth_backup + stepfacD*ph->step);
			}
		}
	}

	sumA /= frameHessians.size();
	sumB /= frameHessians.size();
	sumR /= frameHessians.size();
	sumT /= frameHessians.size();
	sumID /= numID;
	sumNID /= numID;



    if(!setting_debugout_runquiet)
        printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
                sqrtf(sumA) / (0.0005*setting_thOptIterations),
                sqrtf(sumB) / (0.00005*setting_thOptIterations),
                sqrtf(sumR) / (0.00005*setting_thOptIterations),
                sqrtf(sumT)*sumNID / (0.00005*setting_thOptIterations));


	EFDeltaValid=false;
	setPrecalcValues();



	return sqrtf(sumA) < 0.0005*setting_thOptIterations &&
			sqrtf(sumB) < 0.00005*setting_thOptIterations &&
			sqrtf(sumR) < 0.00005*setting_thOptIterations &&
			sqrtf(sumT)*sumNID < 0.00005*setting_thOptIterations;
//
//	printf("mean steps: %f %f %f!\n",
//			meanStepC, meanStepP, meanStepD);
}



// sets linearization point.
void FullSystem::backupState(bool backupLastStep)
{
	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		if(backupLastStep)
		{
			Hcalib.step_backup = Hcalib.step;
			Hcalib.value_backup = Hcalib.value;
			for(FrameHessian* fh : frameHessians)
			{
				fh->step_backup = fh->step;
				fh->state_backup = fh->get_state();
				for(PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup = ph->step;
				}
			}
		}
		else
		{
			Hcalib.step_backup.setZero();
			Hcalib.value_backup = Hcalib.value;
			for(FrameHessian* fh : frameHessians)
			{
				fh->step_backup.setZero();
				fh->state_backup = fh->get_state();
				for(PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup=0;
				}
			}
		}
	}
	else
	{
		Hcalib.value_backup = Hcalib.value;
		for(FrameHessian* fh : frameHessians)
		{
			fh->state_backup = fh->get_state();
			for(PointHessian* ph : fh->pointHessians)
				ph->idepth_backup = ph->idepth;
		}
	}
}

// sets linearization point.
void FullSystem::loadSateBackup()
{
	Hcalib.setValue(Hcalib.value_backup);
	for(FrameHessian* fh : frameHessians)
	{
		fh->setState(fh->state_backup);
		for(PointHessian* ph : fh->pointHessians)
		{
			ph->setIdepth(ph->idepth_backup);

            ph->setIdepthZero(ph->idepth_backup);
		}

	}


	EFDeltaValid=false;
	setPrecalcValues();
}


double FullSystem::calcMEnergy()
{
	if(setting_forceAceptStep) return 0;
	// calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
	//ef->makeIDX();
	//ef->setDeltaF(&Hcalib);
	return ef->calcMEnergyF();

}


void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
	printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
			res[0],
			sqrtf((float)(res[0] / (patternNum*ef->resInA))),
			ef->resInA,
			ef->resInM,
			a,
			b
	);

}


float FullSystem::optimize(int mnumOptIts)
{
    // 如果帧数少于2帧，无法进行优化，直接返回0
    if(frameHessians.size() < 2) return 0;
    
    // 根据帧数动态调整优化迭代次数
    // 帧数越少，需要更多迭代次数来保证收敛
    if(frameHessians.size() < 3) mnumOptIts = 20;  // 3帧以下用20次迭代
    if(frameHessians.size() < 4) mnumOptIts = 15;  // 4帧以下用15次迭代

    // ============= 第一步：收集统计信息和活跃残差 =============
    
    activeResiduals.clear();  // 清空活跃残差列表
    int numPoints = 0;        // 统计点的总数
    int numLRes = 0;          // 统计已线性化残差的数量
    
    // 遍历所有帧中的所有点
    for(FrameHessian* fh : frameHessians)
        for(PointHessian* ph : fh->pointHessians)
        {
            // 遍历每个点的所有残差
            for(PointFrameResidual* r : ph->residuals)
            {
                // 如果残差还未被线性化，则加入活跃残差列表
                if(!r->efResidual->isLinearized)
                {
                    activeResiduals.push_back(r);
                    r->resetOOB();  // 重置越界标志
                }
                else
                    numLRes++;  // 统计已线性化的残差数量
            }
            numPoints++;  // 统计点数
        }

    // 打印优化统计信息：点数、活跃残差数、线性化残差数
    if(!setting_debugout_runquiet)
        printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",ef->nPoints,(int)activeResiduals.size(), numLRes);

    // ============= 第二步：计算初始能量 =============
    
    // 线性化所有残差，计算雅可比矩阵和残差值
    Vec3 lastEnergy = linearizeAll(false);  // false表示不固定线性化点
    double lastEnergyL = calcLEnergy();     // 计算线性化能量项
    double lastEnergyM = calcMEnergy();     // 计算边缘化能量项

    // ============= 第三步：应用残差 =============
    
    // 根据是否启用多线程来应用残差
    if(multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
    else
        applyRes_Reductor(true,0,activeResiduals.size(),0,0);

    // 打印初始误差信息
    if(!setting_debugout_runquiet)
    {
        printf("Initial Error       \t");
        printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
    }

    debugPlotTracking();  // 调试绘图

    // ============= 第四步：主优化循环（Levenberg-Marquardt算法）=============
    
    double lambda = 1e-1;    // LM算法的阻尼因子，初始值0.1
    float stepsize = 1;      // 步长大小
    // 存储上一次的优化变量，用于计算方向变化
    VecX previousX = VecX::Constant(CPARS+ 8*frameHessians.size(), NAN);
    
    // 开始迭代优化
    for(int iteration=0; iteration < mnumOptIts; iteration++)
    {
        // ========== 求解线性系统 ==========
        backupState(iteration!=0);           // 备份当前状态（第一次迭代不备份步长）
        solveSystem(iteration, lambda);      // 求解正规方程组 (H + λI)Δx = -b
        
        // 计算优化方向的变化量（用于自适应步长）
        double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
        previousX = ef->lastX;  // 保存当前的优化变量

        // ========== 自适应步长调整 ==========
        if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
        {
            // 根据方向变化调整步长
            float newStepsize = exp(incDirChange*1.4);
            if(incDirChange<0 && stepsize>1) stepsize=1;  // 方向变化为负且步长>1时，重置为1

            // 使用四次根来平滑步长变化
            stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));
            if(stepsize > 2) stepsize=2;        // 限制最大步长
            if(stepsize <0.25) stepsize=0.25;   // 限制最小步长
        }

        // ========== 执行优化步长 ==========
        // 应用计算出的步长更新到所有优化变量（相机内参、位姿、仿射参数、深度）
        bool canbreak = doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);

        // ========== 计算新的能量 ==========
        Vec3 newEnergy = linearizeAll(false);  // 重新线性化计算新能量
        double newEnergyL = calcLEnergy();     // 新的线性化能量
        double newEnergyM = calcMEnergy();     // 新的边缘化能量

        // 打印当前迭代的优化信息
        if(!setting_debugout_runquiet)
        {
            printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
                // 判断是接受还是拒绝这一步（基于能量是否减少）
                (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
                        lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
                iteration,
                log10(lambda),      // 阻尼因子的对数
                incDirChange,       // 方向变化
                stepsize);          // 当前步长
            printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
        }

        // ========== LM算法：接受或拒绝步长 ==========
        if(setting_forceAceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
                lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
        {
            // 能量减少，接受这一步
            
            // 应用残差更新
            if(multiThreading)
                treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
            else
                applyRes_Reductor(true,0,activeResiduals.size(),0,0);

            // 更新最优能量值
            lastEnergy = newEnergy;
            lastEnergyL = newEnergyL;
            lastEnergyM = newEnergyM;

            lambda *= 0.25;  // 减小阻尼因子（更接近高斯-牛顿法）
        }
        else
        {
            // 能量增加，拒绝这一步
            loadSateBackup();           // 恢复到备份状态
            lastEnergy = linearizeAll(false);  // 重新计算能量
            lastEnergyL = calcLEnergy();
            lastEnergyM = calcMEnergy();
            lambda *= 1e2;             // 增大阻尼因子（更接近梯度下降）
        }

        // 检查收敛条件：如果步长足够小且达到最小迭代次数，则跳出循环
        if(canbreak && iteration >= setting_minOptIterations) break;
    }

    // ============= 第五步：优化后处理 =============
    
    // 设置最后一帧的状态，保持仿射参数不变
    Vec10 newStateZero = Vec10::Zero();
    newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);  // 保持仿射参数

    // 更新最后一帧的评估点
    frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam, newStateZero);
    
    // 重置能量函数相关标志
    EFDeltaValid=false;     // 增量无效，需要重新计算
    EFAdjointsValid=false;  // 伴随矩阵无效，需要重新计算
    
    // 设置伴随矩阵和预计算值
    ef->setAdjointsF(&Hcalib);
    setPrecalcValues();

    // 最终线性化，固定线性化点
    lastEnergy = linearizeAll(true);  // true表示固定线性化点

    // ============= 第六步：检查优化结果 =============
    
    // 检查能量值是否有限，如果无限大则表示跟踪失败
    if(!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
    {
        printf("KF Tracking failed: LOST!\n");
        isLost=true;  // 设置丢失标志
    }

    // 计算最终的RMSE（均方根误差）
    statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

    // ============= 第七步：记录和更新 =============
    
    // 如果启用了标定日志，记录优化结果
    if(calibLog != 0)
    {
        (*calibLog) << Hcalib.value_scaled.transpose() <<           // 相机内参
                " " << frameHessians.back()->get_state_scaled().transpose() <<  // 最后一帧状态
                " " << sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA))) << // RMSE
                " " << ef->resInM << "\n";  // 边缘化残差数
        calibLog->flush();
    }

    // 更新所有帧的位姿到shell中（用于外部访问）
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        for(FrameHessian* fh : frameHessians)
        {
            fh->shell->camToWorld = fh->PRE_camToWorld;  // 相机到世界坐标变换
            fh->shell->aff_g2l = fh->aff_g2l();          // 仿射光度变换参数
        }
    }

    debugPlotTracking();  // 调试绘图

    // 返回最终的RMSE作为优化质量指标
    return sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));
}





void FullSystem::solveSystem(int iteration, double lambda)
{
	ef->lastNullspaces_forLogging = getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

	ef->solveSystemF(iteration, lambda,&Hcalib);
}



double FullSystem::calcLEnergy()
{
	if(setting_forceAceptStep) return 0;

	double Ef = ef->calcLEnergyF_MT();
	return Ef;

}


void FullSystem::removeOutliers()
{
	int numPointsDropped=0;
	for(FrameHessian* fh : frameHessians)
	{
		for(unsigned int i=0;i<fh->pointHessians.size();i++)
		{
			PointHessian* ph = fh->pointHessians[i];
			if(ph==0) continue;

			if(ph->residuals.size() == 0)
			{
				fh->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				fh->pointHessians[i] = fh->pointHessians.back();
				fh->pointHessians.pop_back();
				i--;
				numPointsDropped++;
			}
		}
	}
	ef->dropPointsF();
}




std::vector<VecX> FullSystem::getNullspaces(
		std::vector<VecX> &nullspaces_pose,
		std::vector<VecX> &nullspaces_scale,
		std::vector<VecX> &nullspaces_affA,
		std::vector<VecX> &nullspaces_affB)
{
	nullspaces_pose.clear();
	nullspaces_scale.clear();
	nullspaces_affA.clear();
	nullspaces_affB.clear();


	int n=CPARS+frameHessians.size()*8;
	std::vector<VecX> nullspaces_x0_pre;
	for(int i=0;i<6;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(FrameHessian* fh : frameHessians)
		{
			nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_pose.col(i);
			nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;
			nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		nullspaces_pose.push_back(nullspace_x0);
	}
	for(int i=0;i<2;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(FrameHessian* fh : frameHessians)
		{
			nullspace_x0.segment<2>(CPARS+fh->idx*8+6) = fh->nullspaces_affine.col(i).head<2>();
			nullspace_x0[CPARS+fh->idx*8+6] *= SCALE_A_INVERSE;
			nullspace_x0[CPARS+fh->idx*8+7] *= SCALE_B_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		if(i==0) nullspaces_affA.push_back(nullspace_x0);
		if(i==1) nullspaces_affB.push_back(nullspace_x0);
	}

	VecX nullspace_x0(n);
	nullspace_x0.setZero();
	for(FrameHessian* fh : frameHessians)
	{
		nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_scale;
		nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;
		nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
	}
	nullspaces_x0_pre.push_back(nullspace_x0);
	nullspaces_scale.push_back(nullspace_x0);

	return nullspaces_x0_pre;
}

}
