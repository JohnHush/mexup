pipeline {
    agent none
		stage('部署') {
		    agent {
                node {
                    label 'Jenkins-node1'
                }
            }
			steps {
			    sh 'hostname'
                sh 'pwd'
             //  sh 'scp -o StrictHostKeyChecking=no order-app/target/order-app-1.0-SNAPSHOT.jar centos@10.8.31.215:/application/order/order-app/'
             //  sh 'scp -o StrictHostKeyChecking=no settlement-app/target/settlement-app-1.0-SNAPSHOT.jar centos@10.8.31.215:/application/order/settlement-app/'
            } 
		}
		stage('启动') {
		    agent {
                node {
                    label 'Jenkins-node1'
                }
            }
		 	steps {
		 	    sh 'pwd'
            //    sh 'ssh centos@10.8.31.215 \'bash -c "sh /application/order/order-app/start.sh"\''
            //    sh 'ssh centos@10.8.31.215 \'bash -c "sh /application/order/settlement-app/start.sh"\''
            //    sh 'sleep 60'
            //    sh 'ssh centos@10.8.31.215 \'bash -c "tail -n10000 /application/order/order-app/work/logs/stdout.log"\''
            //    sh 'ssh centos@10.8.31.215 \'bash -c "tail -n10000 /application/order/settlement-app/work/logs/stdout.log"\''
            }
		}
		
    }
}