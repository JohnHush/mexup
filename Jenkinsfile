pipeline {
    agent none
    stages {
        stage('部署') {
            agent {
                node {
                    label 'Jenkins-node1'
                }
            }
            steps {
                sh 'hostname'
                sh 'pwd'
                sh 'scp -r -o StrictHostKeyChecking=no /data/jenkins-slave/workspace/sport-quantization_master/* centos@10.8.24.66:/quantization/sport-quantization-master/'
    
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
                sh 'ssh centos@10.8.24.66 \'bash -c "sh /quantization/sport-quantization-master/start_master.sh"\''
                
            }
        }
    }
}