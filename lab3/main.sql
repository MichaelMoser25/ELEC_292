-- Retrieve all columns from both employees and salaries tables.
SELECT *
FROM employees;

SELECT *
FROM salaries;

-- Find the records whose salary times 1.7 is greater than $100000
SELECT *
FROM salaries
WHERE salary * 1.7 > 100000;

-- How many distinct last_names can be found in employees table?
SELECT COUNT(DISTINCT last_name) FROM employees;

-- Find the average of the salaries whose emp_no is greater than 1510
SELECT AVG(salary)
FROM salaries
WHERE emp_no > 1510;

-- Find the unique last_names and the number of the times these last_names are encountered in our database
SELECT last_name, COUNT(last_name)
FROM employees
GROUP BY last_name;


-- From the salaries table, find the unique emp_no and the average salary of each unique emp_no.
SELECT DISTINCT emp_no, AVG(salary)
FROM salaries
GROUP BY emp_no;

-- From the ‘Employees’ database, find the first_name, last_name, and salary of all the employees.
SELECT first_name, last_name, salary
FROM employees
INNER JOIN salaries
ON employees.emp_no = salaries.emp_no;

-- Write a procedure named ‘emp_avg_salary’. This procedure must accept employee number as
-- its input, and show the average salary associated with this employee number. Call this procedure
-- with the employee number 11300. The result must be 48193.8.
DELIMITER $$
CREATE PROCEDURE emp_avg_salary(IN p_emp_no INT)
BEGIN
	SELECT AVG(salary)
    FROM salaries
    WHERE emp_no = p_emp_no;
END $$
DELIMITER ;

CALL emp_avg_salary(11300);

