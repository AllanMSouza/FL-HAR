const axios = require('axios');

/**
 * @typedef {Object} ServerAccuracy
 * @property {string[]} serverAccuracy.accuracies - Array of accuracies for the server
 * @property {string[]} serverAccuracy.timestamps - Array of timestamps for the server
 */

/**
 * @typedef {Object} ContainerInfo
 * @property {string} containerInfos[].ID - The container ID.
 * @property {string} containerInfos[].host - The node hosting the container.
 * @property {string} containerInfos[].image - The name of the image.
 * @property {string} containerInfos[].state - The state of a container.
 * @property {string} containerInfos[].status - The status of a container.
 * @property {Object[]} containerInfos[].stats - Array of stats of a container.
 */

/**
 * Retrieves an array of container information for every 
 * containers of the given nodes alongside their stats
 * @param {string[]} nodes - List of nodes.
 * @param {string} [port=2375] - Port to access node.
 * @returns {Promise<ContainerInfo[]>} containerInfos - List of container information with stats.
 */
exports.getAllContainersInfosWithStats = async (nodes, port) => {
    const containerInfos = await this.getAllContainersInfos(nodes, port);
    for (const containerInfo of containerInfos) {
        const stats = await this.getContainerStats(containerInfo.host, containerInfo.ID);
        if (containerInfo.image === 'server') {
            const accuracy = await this.getServerAccuracy(containerInfo.ID, containerInfo.host);
            if (accuracy >= 0) {
                stats.accuracy = accuracy;
            }
        }
        containerInfo.stats = [stats];
    }

    return containerInfos;
}

/**
 * Retrieves an array of container information for every containers of the given nodes.
 * The stats are not retrieved, only the current information for the container.
 * @param {string[]} nodes - List of nodes.
 * @param {string} [port=2375] - Port to access node.
 * @returns {Promise<ContainerInfo[]>} containerInfos - List of container information.
 */
exports.getAllContainersInfos = async (nodes, port) => {
    const containerInfos = [];
    for (let node of nodes) {
        const defaultPort = '2375';
        const response = await sendGetRequest(`http://${node}:${port || defaultPort}/containers/json`);
        response.data.forEach(container => {
            containerInfos.push({
                ID: container.Id,
                host: node,
                image: getContainerImageName(container.Image),
                state: container.State,
                status: container.Status,
                stats: []
            });
        });
    }

    return containerInfos;
}

/**
 * Retrieves stats for a given container in a given host.
 * @param {string} id - ID of the container.
 * @param {string} host - Host node of the container.
 * @returns {Promise<Object>} containerStats - Stats of the container.
 */
exports.getContainerStats = async (host, id) => {
    const response = await sendGetRequest(`http://${host}:2375/containers/${id}/stats?stream=false`);
    if (response) {
        return response.data;
    }
}

/**
 * Retrieves accuracy for a given container, taking into account
 * that the container is a server which logs accuracy. If the container
 * cannot be found, returns -1.
 * @param {string} containerId - ID of a container
 * @param {string} containerHost - Host in which container is hosted
 * @returns {Promise<Number>} serverAccuracy - Current accuracy for the server
 */
 exports.getServerAccuracy = async (containerId, containerHost) => {
    const response = await sendGetRequest(`http://${containerHost}:2375/containers/${containerId}/logs?stdout=true&timestamps=true`);
    if (!response) {
        return -1;
    }
    const lines = response.data.split("\n");

    lines.pop(); // Removes last line which is always empty

    const lastLine = lines.slice(-1)[0]

    if (lastLine) {
        return getLastNumberInString(lastLine);
    }

    return 0;
}

const getContainerImageName = (image) => {
    const regex = /(?<=\:)(.*?)(?=\@)/;
    const found = image.match(regex);
    return found[0];
}

const getLastNumberInString = (string) => {
    return string.split(" ").pop();
}

const sendGetRequest = async (url) => {
    try {
        const response = await axios.get(url);
        return response;
    } catch (err) {
        console.error(err);
    }
};
