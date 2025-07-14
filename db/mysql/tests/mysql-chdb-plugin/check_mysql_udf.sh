#!/bin/bash

echo "=== MySQL UDF Capability Check ==="
echo

# Check if plugin directory exists and is accessible
echo "1. Plugin directory:"
ls -la /usr/lib/mysql/plugin/ | head -5

echo
echo "2. MySQL plugin variables:"
mysql teste -u root -pteste -e "SHOW VARIABLES LIKE 'plugin_dir';" 2>/dev/null

echo
echo "3. Check if we can load ANY UDF:"
# Try to load a simple existing UDF
mysql teste -u root -pteste -e "SHOW PLUGINS;" 2>/dev/null | grep -i udf

echo
echo "4. MySQL error log (last 20 lines):"
sudo tail -20 /var/log/mysql/error.log

echo
echo "5. Alternative: Create functions without SONAME (stored functions):"
mysql teste -u root -pteste << 'EOF'
DELIMITER //

DROP FUNCTION IF EXISTS ch_customer_count_stored//

CREATE FUNCTION ch_customer_count_stored() 
RETURNS INT
DETERMINISTIC
READS SQL DATA
BEGIN
    -- This is just a placeholder
    -- In reality, we'd need to call our helper somehow
    RETURN 100;
END//

DELIMITER ;

-- Test it
SELECT ch_customer_count_stored() as count;
EOF