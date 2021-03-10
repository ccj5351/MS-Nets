#t=14400
#echo "Hi, I'm sleeping for $t seconds..."
#sleep ${t}s

#---------------
# utility function
#---------------
function makeDir () {
	dstDir="$1"
	if [ ! -d $dstDir ]; then
		mkdir -p $dstDir
		echo "mkdir $dstDir"
	else
		echo $dstDir exists
	fi
}  

DATA_ROOT="/media/ccjData2"
if [ ! -d $DATA_ROOT ];then
	DATA_ROOT="/data/ccjData"
	echo "Updated : setting data_root = ${DATA_ROOT}"
fi


#----------------------------
#--- PROJECT_ROOT -----------
#----------------------------
PROJECT_ROOT="~/ccj-papers-github-codes/MS-Nets"

#----------------------------
#--- DATA TYPES -------------
#----------------------------
KT2012=0 KT2015=1 ETH3D=0 MIDDLEBURY=0

if [ $KT2012 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
  TRAINING_LIST="lists/kitti2012_train170.list"
  TEST_LIST="lists/kitti2012_val24.list"
	#Note that the “crop_width” and “crop_height” must be multiple of 48, 
	#"max_disp" must be multiple of 12 (default: 192).
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192

elif [ $KT2015 -eq 1 ]; then
	DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
	TRAINING_LIST="lists/kitti2015_train170.list"
  TEST_LIST="lists/kitti2015_val30.list"
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
	#let MAX_DISP=180

else
	DATA_PATH="${DATA_ROOT}/datasets/SceneFlowDataset/"
  TRAINING_LIST="lists/sceneflow_train.list"
  #TRAINING_LIST="lists/sceneflow_train_small.list"
  #TRAINING_LIST="lists/sceneflow_train_small_500.list"
  #TEST_LIST="lists/sceneflow_test_select.list"
  TEST_LIST="lists/sceneflow_test_small.list"
	#let CROP_HEIGHT=256
	#let CROP_WIDTH=512
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
fi
echo "DATA_PATH=$DATA_PATH"

START_EPOCH=0
#NUM_EPOCHS=400
NUM_EPOCHS=10
#NUM_EPOCHS=30
NUM_WORKERS=8
BATCHSIZE=2

LOG_SUMMARY_STEP=40
#LOG_SUMMARY_STEP=200

#----------------------------
#--- TASK TYPES -------------
#----------------------------
#TASK_TYPE='train'
#TASK_TYPE='loop-train'
#TASK_TYPE='val-30'
TASK_TYPE='cross-val'
#TASK_TYPE='eval-badx'
#############################

#----------------------------
#--- MODELTYPES -------------
#----------------------------
MODEL_NAME='MS-GCNet'
#MODEL_NAME='MS-PSMNet'

if [ $MODEL_NAME == 'MS-GCNet' ]; then
	MODEL_NAME_STR="msgcnet"
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
	LOG_SUMMARY_STEP=50
  if [ $KT2012 -eq 1 ]; then
		RESUME='./checkpoints/saved/msgcnet-pretrained/pretrained_sceneflow_epoch_00010.tar'
    EXP_NAME="msgcnet-D${MAX_DISP}-sfepo10-kt12epo${NUM_EPOCHS}"
  elif [ $KT2015 -eq 1 ]; then
		RESUME='./checkpoints/saved/msgcnet-pretrained/pretrained_sceneflow_epoch_00010.tar'
    EXP_NAME="msgcnet-D${MAX_DISP}-sfepo10-kt15epo${NUM_EPOCHS}"
	else
		RESUME=''
    EXP_NAME="msgcnet-D${MAX_DISP}-sfepo${NUM_EPOCHS}"
	fi


elif [ $MODEL_NAME == 'MS-PSMNet' ]; then
	MODEL_NAME_STR="mspsmnet"
	let CROP_HEIGHT=256
	let CROP_WIDTH=512
	let MAX_DISP=192
  if [ $KT2012 -eq 1 ]; then
		RESUME='./checkpoints/saved/mspsmnet-pretrained/pretrained_sceneflow_epoch_00010.tar'
    EXP_NAME="mspsmnet-D${MAX_DISP}-sfepo10-kt12epo${NUM_EPOCHS}"
  elif [ $KT2015 -eq 1 ]; then
		RESUME='./checkpoints/saved/mspsmnet-pretrained/pretrained_sceneflow_epoch_00010.tar'
    EXP_NAME="mspsmnet-D${MAX_DISP}-sfepo10-kt15epo${NUM_EPOCHS}"
	else
		RESUME=''
    EXP_NAME="mspsmnet-D${MAX_DISP}-sfepo${NUM_EPOCHS}"
	fi
fi

TRAIN_LOGDIR="./logs/${EXP_NAME}"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
echo "EXP_NAME=$EXP_NAME"
echo "TRAIN_LOGDIR=$TRAIN_LOGDIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
#exit


################################################################
# Netwrok Training SF training, 
# for-loop due to MS feature stuck at epoch 1 !!!
################################################################
#TASK_TYPE='loop-train'
if [ "$TASK_TYPE" = 'loop-train' ]; then
	flag=true
else
	flag=false
fi
echo "TASK_TYPE=$TASK_TYPE, flag=$flag"
if [ "$flag" = true ]; then
	MODE='train'
	let END_EPOCH=${NUM_EPOCHS}-1
	echo "END_EPOCH=$END_EPOCH"
	
	for epo_idx in $(seq ${START_EPOCH} ${END_EPOCH})
	do
    if [ $epo_idx -eq 0 ]; then
			echo "Using ${RESUME}"
		elif [ $epo_idx -eq ${START_EPOCH} ]; then
			let epo_model=${epo_idx}
			RESUME="./checkpoints/saved/${EXP_NAME}/${MODEL_NAME}/model_epoch_$(printf "%05d" "$epo_model").tar"
		else
			let epo_model=${epo_idx}
			RESUME="./checkpoints/${EXP_NAME}/${MODEL_NAME}/model_epoch_$(printf "%05d" "$epo_model").tar"
		fi
		echo "Loop training : RESUME=$RESUME"
    echo "EXP_NAME=$EXP_NAME"
		CUDA_VISIBLE_DEVICES=0 python3.7 -m main_msnet \
			--batchSize=${BATCHSIZE} \
			--crop_height=$CROP_HEIGHT \
			--crop_width=$CROP_WIDTH \
			--max_disp=$MAX_DISP \
			--train_logdir=$TRAIN_LOGDIR \
			--thread=${NUM_WORKERS} \
			--data_path=$DATA_PATH \
			--training_list=$TRAINING_LIST \
			--test_list=$TEST_LIST \
			--checkpoint_dir=$CHECKPOINT_DIR \
			--log_summary_step=${LOG_SUMMARY_STEP} \
			--resume=$RESUME \
			--nEpochs=1 \
			--startEpoch=$epo_idx \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
			--eth3d=$ETH3D \
			--middlebury=$MIDDLEBURY \
			--mode=$MODE \
			--resultDir=$RESULTDIR \
			--model_name=$MODEL_NAME \
		  --sf_frame=$SF_FRAME
	done
  exit
fi

################################
# Netwrok Training & profiling
################################
if [ "$TASK_TYPE" = 'train' ]; then
	flag=true
else
	flag=false
fi
echo "TASK_TYPE=$TASK_TYPE, flag=$flag"
if [ "$flag" = true ]; then
	MODE='train'
  RESULTDIR="./results/${EXP_NAME}"
	CUDA_VISIBLE_DEVICES=0 python3.7 -m main_msnet \
		--batchSize=${BATCHSIZE} \
		--crop_height=$CROP_HEIGHT \
		--crop_width=$CROP_WIDTH \
		--max_disp=$MAX_DISP \
		--train_logdir=$TRAIN_LOGDIR \
		--thread=${NUM_WORKERS} \
		--data_path=$DATA_PATH \
		--training_list=$TRAINING_LIST \
		--test_list=$TEST_LIST \
		--checkpoint_dir=$CHECKPOINT_DIR \
		--log_summary_step=${LOG_SUMMARY_STEP} \
		--resume=$RESUME \
		--nEpochs=$NUM_EPOCHS \
		--startEpoch=$START_EPOCH \
		--kitti2012=$KT2012 \
		--kitti2015=$KT2015 \
		--eth3d=$ETH3D \
		--middlebury=$MIDDLEBURY \
		--mode=$MODE \
		--resultDir=$RESULTDIR \
		--model_name=$MODEL_NAME
	  exit
fi

####################################
# Netwrok validation on val dataset 
####################################
KT2012=0 KT2015=0 ETH3D=0 MIDDLEBURY=0
if [ "$TASK_TYPE" = 'val-30' ]; then
	flag=true
else
	flag=false
fi
if [ "$flag" = true ]; then
	MODE='test'
	#KT15/12: crop_height=384, crop_width=1248
	#sceneflow: crop_height=576, crop_width=960
	#let CROP_HEIGHT=384
	#let CROP_WIDTH=1248
	let MAX_DISP=192

	if [ $KT2012 -eq 1 ] || [ $KT2015 -eq 1 ]; then
		
		let CROP_HEIGHT=384
		let CROP_WIDTH=1248 # multiple of 32, due to the Architecture;

	else
		#let CROP_HEIGHT=576
		#let CROP_WIDTH=960
		let CROP_HEIGHT=256
		let CROP_WIDTH=512
  fi	
		
	declare -a ALL_EPOS_TEST=(30 25 20 15 10 5)
	SF_FRAME='frames_finalpass'
	#SF_FRAME='frames_cleanpass'
	for idx in $(seq 0 0)
	#for idx in $(seq 1 15)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo "Testing using model at epoch = $EPO_TEST"
		
		TMP_MODEL_NAME="${MODEL_NAME_STR}-D192-sfepo30"
		RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/${MODEL_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
		if [ "$1" == 0 ]; then
			if [ "$KT2015" == 1 ]; then
				echo "test ${MODEL_NAME}: SF --> KT15 !!!"
	            DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
				KT2012=0
				TEST_LIST="lists/kitti2015_val30.list"
				EXP_NAME="${TMP_MODEL_NAME}-testKT15/disp-epo-$(printf "%03d" "$EPO_TEST")"
			elif [ "$KT2012" == 1 ]; then
	            DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
				echo "test ${MODEL_NAME}: SF --> KT12 !!!"
				KT2015=0
				TEST_LIST="lists/kitti2012_val24.list"
				EXP_NAME="${TMP_MODEL_NAME}-testKT12/disp-epo-$(printf "%03d" "$EPO_TEST")"
			else
				echo "test ${MODEL_NAME}: SF --> SF Val-2k validation set!!!"
	            DATA_PATH="${DATA_ROOT}/datasets/SceneFlowDataset/"
				KT2012=0
				KT2015=0
				#TEST_LIST="lists/sceneflow_val.list"
				TEST_LIST="lists/sceneflow_val_small.list"
				EXP_NAME="${TMP_MODEL_NAME}-sfVal2k/disp-epo-$(printf "%03d" "$EPO_TEST")"
			fi
		
		else 
			echo "You have to specify a argument to bash!!!"
			exit
		fi
		
		cd ${PROJECT_ROOT}
		RESULTDIR="./results/${EXP_NAME}"
	  CUDA_VISIBLE_DEVICES=0 python3.7 -m main_msnet \
			--batchSize=${BATCHSIZE} \
			--crop_height=$CROP_HEIGHT \
			--crop_width=$CROP_WIDTH \
			--max_disp=$MAX_DISP \
			--train_logdir=$TRAIN_LOGDIR \
			--thread=${NUM_WORKERS} \
			--data_path=$DATA_PATH \
			--training_list=$TRAINING_LIST \
			--test_list=$TEST_LIST \
			--checkpoint_dir=$CHECKPOINT_DIR \
			--log_summary_step=${LOG_SUMMARY_STEP} \
			--resume=$RESUME \
			--nEpochs=$NUM_EPOCHS \
			--startEpoch=$START_EPOCH \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
			--eth3d=$ETH3D \
			--middlebury=$MIDDLEBURY \
			--mode=$MODE \
			--resultDir=$RESULTDIR \
			--model_name=$MODEL_NAME \
		  --sf_frame=$SF_FRAME
		
		if [ $KT2015 -eq 1 ]; then
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/val-30"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-30"
			done	
	  elif [ $KT2012 -eq 1 ]; then
			# move pfm files to val-30 subdir
			makeDir "$RESULTDIR/val-24"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./val-24"
			done
		fi

	done
fi # end of Netwrok Testing


###################################
# MSNet-Generalization experiments 
###################################

#KT2012=1 KT2015=0 ETH3D=0 MIDDLEBURY=0
KT2012=0 KT2015=1 ETH3D=0 MIDDLEBURY=0
#KT2012=0 KT2015=0 ETH3D=1 MIDDLEBURY=0
#KT2012=0 KT2015=0 ETH3D=0 MIDDLEBURY=1

if [ "$TASK_TYPE" = 'cross-val' ]; then
	flag=true
	MODE='test'
elif [ "$TASK_TYPE" = 'eval-badx' ]; then
	flag=true
	MODE='eval-badx'
else
	flag=false
fi
echo "TASK_TYPE=$TASK_TYPE, flag=$flag"

if [ "$flag" = true ]; then
	#KT15/12: crop_height=384, crop_width=1248
	#sceneflow: crop_height=576, crop_width=960
	let MAX_DISP=192
	
	if [ $KT2012 -eq 1 ] || [ $KT2015 -eq 1 ]; then
		let CROP_HEIGHT=384
		let CROP_WIDTH=1248

	else
		let CROP_HEIGHT=576
		let CROP_WIDTH=960
  fi


	declare -a ALL_EPOS_TEST=(30 10)
	for idx in $(seq 0 0)	
	#for idx in $(seq 0 15)
	do # epoch model loop
		EPO_TEST=${ALL_EPOS_TEST[idx]}
		echo "Testing using model at epoch = $EPO_TEST"
		#-------------------------------
		# baseline 1: pre-trained MS-GCNet:
		#-------------------------------
	  if [ "$1" == 8 ]; then
			#echo "test GCNet baseline: SF --> KITTI !!!"
      TMP_MODEL_NAME="${MODEL_NAME_STR}-D${MAX_DISP}-sfepo30"
			#RESUME="./checkpoints/saved/${TMP_MODEL_NAME}/${MODEL_NAME}/model_epoch_$(printf "%05d" "$EPO_TEST").tar"
			RESUME="./checkpoints/saved/msgcnet-pretrained/pretrained_sceneflow_epoch_$(printf "%05d" "$EPO_TEST").tar"
			if [ "$KT2015" == 1 ]; then
				echo "test KT15 train-200 !!!"
				DATA_PATH="${DATA_ROOT}/datasets/KITTI-2015/training/"
				KT2012=0
        ETH3D=0 
				MIDDLEBURY=0
				TEST_LIST="lists/kitti2015.list"
				#TEST_LIST="lists/kitti2015_tmp_small.list"
				EXP_NAME="${TMP_MODEL_NAME}-testKT15/disp-epo-$(printf "%03d" "$EPO_TEST")"
			elif [ "$KT2012" == 1 ]; then
				echo "test KT12 train-194 !!!"
				KT2015=0
        ETH3D=0 
				MIDDLEBURY=0
				DATA_PATH="${DATA_ROOT}/datasets/KITTI-2012/training/"
				TEST_LIST="lists/kitti2012.list"
				EXP_NAME="${TMP_MODEL_NAME}-testKT12/disp-epo-$(printf "%03d" "$EPO_TEST")"
			elif [ "$ETH3D" == 1 ]; then
				echo "test ETH3D train-27 !!!"
				KT2012=0
				KT2015=0
				MIDDLEBURY=0
				DATA_PATH="${DATA_ROOT}/datasets/ETH3D/two_view_training/"
				TEST_LIST="lists/eth3d_train.list"
				EXP_NAME="${TMP_MODEL_NAME}-testETH3D/disp-epo-$(printf "%03d" "$EPO_TEST")"
			elif [ "$MIDDLEBURY" == 1 ]; then
				echo "test Middlebury train-15 !!!"
				KT2012=0
				KT2015=0
        ETH3D=0 
				DATA_PATH="${DATA_ROOT}/datasets/MiddleBury/MiddEval3/trainingH/"
				#DATA_PATH="${DATA_ROOT}/datasets/MiddleBury/MiddEval3/trainingQ/"
				TEST_LIST="lists/middleburyV3H_train.list"
				EXP_NAME="${TMP_MODEL_NAME}-testMB14/disp-epo-$(printf "%03d" "$EPO_TEST")"
				#EXP_NAME="${TMP_MODEL_NAME}-testMB14Q/disp-epo-$(printf "%03d" "$EPO_TEST")"
			fi
		
		else 
			echo "You have to specify a argument to bash!!!"
			exit
		fi
		
		RESULTDIR="./results/${EXP_NAME}"
		cd ${PROJECT_ROOT}
	  CUDA_VISIBLE_DEVICES=0 python3.7 -m main_msnet \
			--batchSize=${BATCHSIZE} \
			--crop_height=$CROP_HEIGHT \
			--crop_width=$CROP_WIDTH \
			--max_disp=$MAX_DISP \
			--train_logdir=$TRAIN_LOGDIR \
			--thread=${NUM_WORKERS} \
			--data_path=$DATA_PATH \
			--training_list=$TRAINING_LIST \
			--test_list=$TEST_LIST \
			--checkpoint_dir=$CHECKPOINT_DIR \
			--log_summary_step=${LOG_SUMMARY_STEP} \
			--resume=$RESUME \
			--nEpochs=$NUM_EPOCHS \
			--startEpoch=$START_EPOCH \
			--kitti2012=$KT2012 \
			--kitti2015=$KT2015 \
			--eth3d=$ETH3D \
			--middlebury=$MIDDLEBURY \
			--mode=$MODE \
			--resultDir=$RESULTDIR \
			--model_name=$MODEL_NAME \
		  --sf_frame=$SF_FRAME
		
		
    if [ "$TASK_TYPE" = 'eval-badx' ]; then
			exit
		fi

		if [ $KT2015 -eq 1 ] || [ $KT2012 -eq 1 ] || [ $ETH3D -eq 1 ] || [ $MIDDLEBURY -eq 1 ]; then
			# move pfm files to disp-pfm subdir
			makeDir "$RESULTDIR/disp-pfm"
			cd $RESULTDIR
			for i in *.pfm; do
				mv -i -- "$i" "./disp-pfm"
			done	
		fi

	done
fi # end of Netwrok Benchmark Submission
