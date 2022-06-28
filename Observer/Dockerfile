FROM node:12.18.1
ADD src /src
ADD package.json /
ADD package-lock.json /
RUN npm install
CMD [ "node","./src/observer.js" ]
