const dockerService = require('./docker-service');
fs = require('fs');


const WORKER_01       = '192.168.56.100' //process.env.WORKER_01.toString();
const WORKER_02       = '192.168.56.101' //process.env.WORKER_02.toString();
const WORKER_03       = '192.168.56.102' //process.env.WORKER_03.toString();
const MANAGER         = '192.168.56.103' //process.env.MANAGER.toString();
const EXPERIMENT_NAME = 'NormalFL'//process.env.EXPERIMENT_NAME.toString();

const NODES = [WORKER_01, WORKER_02, WORKER_03, MANAGER];


observe = async () => {
    let containerInfos = await dockerService.getAllContainersInfosWithStats(NODES);

    while(anyContainerIsRunning(containerInfos)) {
        const containerInfosIds = containerInfos.map(containerInfo => containerInfo.ID);
        const updatedContainerInfos = await dockerService.getAllContainersInfosWithStats(NODES);
        const updatedContainerInfosIds = updatedContainerInfos.map(updatedContainerInfo => updatedContainerInfo.ID);

        updatedContainerInfos.forEach(updatedContainerInfo => {
            // If container already retrieved, only update the State,
            // Status, and push the retrieved Stats to the array.
            // Else it's a new container, then add it to the containerInfos list.
            if (containerInfosIds.includes(updatedContainerInfo.ID)) {
                containerInfos.forEach(containerInfo => {
                    if (containerInfo.ID === updatedContainerInfo.ID) {
                        containerInfo.state = updatedContainerInfo.state;
                        containerInfo.status = updatedContainerInfo.status;
                        containerInfo.stats.push(updatedContainerInfo.stats[0]);
                    }
                });
            } else {
                containerInfos.push(updatedContainerInfo);
            }
        });

        // If there is a container in containerInfo which hasn't been retrieved,
        // it means the container stopped. Therefore, we can persist its stats
        // and remove it from the list
        for (const [index, containerInfo] of containerInfos.entries()) {
            if (!updatedContainerInfosIds.includes(containerInfo.ID)) {
                const currentTime = new Date().toISOString();
                const targetDir = `Reports/${EXPERIMENT_NAME}/${containerInfo.image}`;
                fs.mkdirSync(targetDir, { recursive: true });
                fs.writeFileSync(`${targetDir}/${currentTime}-${containerInfo.host}`, JSON.stringify(containerInfo.stats));
                containerInfos.splice(index, 1);
            }
        }
    }
}

const anyContainerIsRunning = (containerInfos) => {
    return containerInfos.some(containerInfo => containerInfo.state === 'running');
}

module.exports = observe();
