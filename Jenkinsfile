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
            }
        }
    }
}