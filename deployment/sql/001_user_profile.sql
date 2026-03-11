-- User profile bootstrap (minimal model: ADMIN / USER)

BEGIN
    EXECUTE IMMEDIATE q'[
        CREATE TABLE AIUSER.USER_PROFILE (
            USERNAME      VARCHAR2(128) PRIMARY KEY,
            PROFILE_CODE  VARCHAR2(16) NOT NULL,
            ENABLED       NUMBER(1) DEFAULT 1 NOT NULL,
            UPDATED_AT    TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT CK_USER_PROFILE_CODE CHECK (PROFILE_CODE IN ('ADMIN', 'USER')),
            CONSTRAINT CK_USER_PROFILE_ENABLED CHECK (ENABLED IN (0, 1))
        )
    ]';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;
/

MERGE INTO AIUSER.USER_PROFILE t
USING (
    SELECT 'luigi' AS username, 'ADMIN' AS profile_code, 1 AS enabled
    FROM dual
) s
ON (UPPER(t.username) = UPPER(s.username))
WHEN MATCHED THEN
    UPDATE SET
        t.profile_code = s.profile_code,
        t.enabled = s.enabled,
        t.updated_at = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (username, profile_code, enabled, updated_at)
    VALUES (s.username, s.profile_code, s.enabled, SYSTIMESTAMP);

COMMIT;
