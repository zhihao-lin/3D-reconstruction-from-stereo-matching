const express = require('express');
const app = express()
const http = require('http');
const server = http.createServer(app)
const cors = require('cors');


// import app routers
app.use(cors({credentials: true, origin: true}));
app.use('/objs', express.static(__dirname+ "/objs"));
app.get('/', (req, res)=>{
    console.log('into')
    res.json("hi").status(200)
});

const port = 8080||process.env.Port;
server.listen(port, ()=>{console.log(`listening to port ${port}`)});
