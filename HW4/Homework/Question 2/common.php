<?php
function sql($query)
{
    static $connection=null;

    $args=func_get_args();
    array_shift ( $args );
    if ($connection===null) // First time connect
    {
        $connection=new PDO($query,@$args[0],@$args[1]);
        $connection->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        return $connection;
    }
    $statement = $connection->prepare ( $query );
    $out=$statement->execute ($args);
    $type = substr ( trim ( strtoupper ( $query ) ), 0, 6 );
    if ($type == "INSERT")
    {
        $res = $connection->lastInsertId(); // returns 0 if no auto-increment found
        if ($res == 0)
            $res = $statement->rowCount ();
        return $res;
    }
    elseif ($type == "DELETE" or $type == "UPDATE" or $type == "REPLAC")
        return $statement->rowCount ();
    elseif ($type == "SELECT")
    {
        $res=[];
        while ($r=$statement->fetchObject())
            $res[]=$r;
        return $res;
    }
    else
        return $out;
}
function setup($connection_string, $user=null, $pass=null)
{
    sql($connection_string, $user, $pass);
    sql("DROP TABLE IF EXISTS users");
    sql("CREATE TABLE IF NOT EXISTS users
        (id integer primary key AUTOINCREMENT, username text, password text)");

    if (!sql("SELECT * FROM users"))
    {
        sql("INSERT INTO users (username,password) VALUES ('jack',?)", md5(rand()));
        sql("INSERT INTO users (username,password) VALUES ('admin',?)", md5(rand()));
        sql("INSERT INTO users (username,password) VALUES ('lord',?)", md5(rand()));
        sql("INSERT INTO users (username,password) VALUES ('alex',?)", md5(rand()));
        sql("INSERT INTO users (username,password) VALUES ('karen',?)", md5(rand()));
    }

    sql("CREATE TABLE IF NOT EXISTS salaries
        (id integer primary key AUTOINCREMENT,
            userid integer, role text, salary integer, bio text, age integer)");
    if (!sql("SELECT * FROM salaries"))
    {
        sql("INSERT INTO salaries (userid,role,salary,bio,age) VALUES (?,?,?,?,?)",
            1, "assistant", 10000, "Jack is a hard-working assistant!", 32);
        sql("INSERT INTO salaries (userid,role,salary,bio,age) VALUES (?,?,?,?,?)",
            2, "sysadmin", 20000, "Admin manages our systems effectively.", 52);
        sql("INSERT INTO salaries (userid,role,salary,bio,age) VALUES (?,?,?,?,?)",
            3, "boss", 30000, "The boss of the company!", 34);
        sql("INSERT INTO salaries (userid,role,salary,bio,age) VALUES (?,?,?,?,?)",
            4, "intern", 2000, "Highly motivated intern!", 22);
        sql("INSERT INTO salaries (userid,role,salary,bio,age) VALUES (?,?,?,?,?)",
            5, "ceo", 40000, "Best ceo ever!", 48);
    }


}
function input($param)
{
    global $argv;
    static $args=null;
    if ($args===null and !empty($argv))
        parse_str(@$argv[1],$args);
    if (isset($_GET[$param]))
        return $_GET[$param];
    elseif (isset($_POST[$param]))
        return $_POST[$param];
    elseif (isset($args[$param]))
        return $args[$param];
    else
        return null;
}
setup("sqlite:db.sdb");
// setup("mysql:host=localhost;dbname=test;", "root", "root");
