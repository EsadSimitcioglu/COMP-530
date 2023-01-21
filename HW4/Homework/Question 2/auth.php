<?php
include "common.php";
if (isset($_GET['source']))
    die(highlight_string(file_get_contents(__FILE__)));

// Select challenge based on URL
$challenge=input("challenge")*1;
if ($challenge<1 or $challenge>=10)
    $challenge=1;
$index=0;
// If input is provided
if ($username=input("username") and $password=input("password"))
{
    // List of challenges
    if ($challenge==++$index)
    {
        $query="SELECT * FROM users WHERE username='{$username}' AND password='{$password}'";
    }
    
    elseif ($challenge==++$index)
    {
        function escape($in)
        {
            $in=str_replace("'", "\'", $in);
            $in=str_replace('"', '\"', $in);
            return $in;
        }
        // Escape quote characters to disable breaking out
        $username=escape($username);
        $password=escape($password);
        $query="SELECT * FROM users WHERE username='{$username}' AND password='{$password}'";
    }
    
    elseif ($challenge==++$index)
    {
        // Prepared statements
        $ord=input("ord")?:"ORDER BY 1";
        $query="SELECT * FROM users WHERE username=? AND password = ? {$ord}";
        $args=[$username,$password];
    }
    
        
    // Run the query
    try {
        if (!isset($args))
            $r=sql($query);
        elseif (count($args)==1)
            $r=sql($query,$args[0]);
        elseif (count($args)==2)
            $r=sql($query,$args[0],$args[1]);
        elseif (count($args)==3)
            $r=sql($query,$args[0],$args[1],$args[2]);
    }
    catch (PDOException $e)
    {
        $err=$e->getMessage();
    }
    // Debug output
    echo "<pre style='background-color:#EEE;border:1px solid gray;padding:10px;'>Query : ", htmlspecialchars($query);
    echo PHP_EOL,str_repeat("-",80),PHP_EOL;
    if (isset($err))
        echo "Error: ", $err;
    else
        echo "Result: ", print_r($r,true);
    echo "</pre>";

    // Login results
    if (@$r)
        echo "<div style='border:1px dotted darkgreen;background-color:#DFD;padding:5px;'>
                Login successful! Welcome {$r[0]->username}. <a href='?challenge=",($challenge+1),"'>Next Challenge</a></div>";
    else
        echo "<div style='border:1px dotted red;background-color:#FDD;padding:5px;'>
                Invalid username or password</div>";
}
// HTML form follows
?>
<h1>Challenge <?=$challenge?> - Fight!</h1>
<p>
Enter username and password:
</p>
<form method='post'>
<label>Username:</label> <input type='text' name='username'/><br/>
<label>Password:</label> <input type='password' name='password'/><br/>
<input type='submit' />
</form>


<hr/>
<a href='?source=1' target="_blank">Source Code</a>
|
<a href='./'>Back</a>




