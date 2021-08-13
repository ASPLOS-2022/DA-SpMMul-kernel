#! /bin/bash

# if [ $SUITESPARSE_DIR ] then

# fi

LOG_FILE=./log.csv
echo "filename,KDense,SR_RB_RM,PR_RB_RM,SR_EB_RM,PR_EB_RM,SR_RB_CM,PR_RB_CM,SR_EB_CM,PR_EB_CM,cusparse_default,cusparse_1,cusparse_2,cusparse_3,cusparse_4,cusparse_5,cusparse_6,cusparse_7,cusparse_8,cusparse_9" > $LOG_FILE

for i in $@
do
./spmvspmm $i | tee -a $LOG_FILE
done
