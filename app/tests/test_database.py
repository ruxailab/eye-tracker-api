from unittest.mock import patch, MagicMock
from services.database import create_document, get_documents, get_document, delete_document, update_document


@patch('services.database.firestore.client')
def test_create_document(mock_firestore_client):
    mock_db = MagicMock()
    mock_firestore_client.return_value = mock_db
    collection, doc_id, data = 'test_collection', 'test_doc_id', {
        'field': 'value'}
    result = create_document(collection, doc_id, data)
    mock_db.collection().document().set.assert_called_with(data)
    assert result == mock_db.collection().document().set.return_value


@patch('services.database.firestore.client')
def test_get_documents(mock_firestore_client):
    mock_db = MagicMock()
    mock_firestore_client.return_value = mock_db
    collection, field, op, value = 'test_collection', 'field', '==', 'value'
    result = get_documents(collection, field, op, value)
    mock_db.collection().where().stream.assert_called_once()
    assert result == mock_db.collection().where().stream.return_value


@patch('services.database.firestore.client')
def test_get_document(mock_firestore_client):
    mock_db = MagicMock()
    mock_firestore_client.return_value = mock_db
    collection, doc_id = 'test_collection', 'test_doc_id'
    result = get_document(collection, doc_id)
    mock_db.collection().document().get.assert_called_once()
    assert result == mock_db.collection().document().get.return_value


@patch('services.database.firestore.client')
def test_delete_document(mock_firestore_client):
    mock_db = MagicMock()
    mock_firestore_client.return_value = mock_db
    collection, doc_id = 'test_collection', 'test_doc_id'
    result = delete_document(collection, doc_id)
    mock_db.collection().document().delete.assert_called_once()
    assert result == mock_db.collection().document().delete.return_value


@patch('services.database.firestore.client')
def test_update_document(mock_firestore_client):
    mock_db = MagicMock()
    mock_firestore_client.return_value = mock_db
    collection, doc_id, data = 'test_collection', 'test_doc_id', {
        'field': 'new_value'}
    result = update_document(collection, doc_id, data)
    mock_db.collection().document().update.assert_called_with(data)
    assert result == mock_db.collection().document().update.return_value
