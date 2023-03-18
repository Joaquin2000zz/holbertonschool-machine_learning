--
-- Write a SQL script that creates a stored procedure ComputeAverageScoreForUser
--  that computes and store the average score for a student.
-- 
-- Requirements:
-- 
-- - Procedure ComputeAverageScoreForUser is taking 1 input:
-- - user_id, a users.id value (you can assume user_id is linked to an existing users)
--
DROP FUNCTION IF EXISTS `ComputeAverageScoreForUser`;
DELIMITER $$
CREATE PROCEDURE IF NOT EXISTS `ComputeAverageScoreForUser` (IN `user_id` INT)
BEGIN
    DECLARE `mean` INT;
    SELECT AVG(score) INTO `mean`
    FROM `corrections` WHERE (`user_id` = `user_id`);
    UPDATE `users`
    SET
        `average_score` = `mean`
    WHERE
        `id` = `user_id`;
END$$
DELIMITER ;
