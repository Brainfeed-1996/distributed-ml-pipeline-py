export default function handler(req, res) { res.status(200).json({ status: 'ok', service: 'distributed-ml-pipeline-py', timestamp: new Date().toISOString() }); }
