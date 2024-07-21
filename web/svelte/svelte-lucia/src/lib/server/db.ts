import mysql from "mysql2/promise";

export const db = mysql.createPool({
    host: 'localhost', // or the IP address or hostname of the database server
    user: 'root',
    password: '07farm',
    database: 'lucia',
    // Determines whether the pool will queue up connection requests and call them when connections become available, if all connections are in use.
    waitForConnections: true,
     // The maximum number of connections to create at once. 
    connectionLimit: 10,
    // The maximum number of connection requests the pool will queue before returning an error from getConnection. If set to 0, there is no limit to the number of queued connection requests. 
    queueLimit: 0 
});
  
export interface DatabaseUser {
	id: string;
	username: string;
	password: string;
}