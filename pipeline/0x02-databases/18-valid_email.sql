--
-- script that creates a trigger that resets the attribute
-- valid_email only when the email has been changed.
--
-- Context: Nothing related to MySQL, but perfect for
--         user email validation - distribute the logic
--         to the database itself!
--
DROP TRIGGER IF EXISTS `validate`;
CREATE TRIGGER IF NOT EXISTS `validate`
BEFORE UPDATE ON `users`
FOR EACH ROW
SET NEW.`valid_email` = IF(NEW.`email` <> OLD.`email`, 0, NEW.`valid_email`);
