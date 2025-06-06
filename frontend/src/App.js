import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL;

function App() {
  const [boards, setBoards] = useState([]);
  const [sprints, setSprints] = useState([]);
  const [selectedBoard, setSelectedBoard] = useState('');
  const [selectedSprint, setSelectedSprint] = useState('');
  const [excelFile, setExcelFile] = useState(null);
  const [recipientEmail, setRecipientEmail] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingBoards, setLoadingBoards] = useState(true);
  const [loadingSprints, setLoadingSprints] = useState(false);

  useEffect(() => {
    // Fetch boards on component mount
    setLoadingBoards(true);
    axios.get(`${API_URL}/boards`)
      .then(response => {
        setBoards(response.data);
        setLoadingBoards(false);
      })
      .catch(error => {
        console.error('Error fetching boards:', error);
        setMessage('Error loading boards. Please try again.');
        setLoadingBoards(false);
      });
  }, []);

  useEffect(() => {
    if (selectedBoard) {
      // Fetch sprints when a board is selected
      setLoadingSprints(true);
      axios.get(`${API_URL}/sprints?board_id=${selectedBoard}`)
        .then(response => {
          setSprints(response.data);
          setLoadingSprints(false);
        })
        .catch(error => {
          console.error('Error fetching sprints:', error);
          setMessage('Error loading sprints. Please try again.');
          setLoadingSprints(false);
        });
    }
  }, [selectedBoard]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' || 
          file.type === 'application/vnd.ms-excel') {
        setExcelFile(file);
        setMessage('');
      } else {
        setMessage('Please upload a valid Excel file (.xlsx or .xls)');
        event.target.value = null;
      }
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedBoard || !selectedSprint || !excelFile || !recipientEmail) {
      setMessage('Please fill in all fields');
      return;
    }

    setLoading(true);
    setMessage('');

    const formData = new FormData();
    formData.append('board_id', selectedBoard);
    formData.append('sprint_id', selectedSprint);
    formData.append('excel_file', excelFile);
    formData.append('recipient_email', recipientEmail);

    try {
      const response = await axios.post(`${API_URL}/generate_report`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setMessage(response.data.message || 'Report generated and email sent successfully');
      // Reset form after successful submission
      setSelectedSprint('');
      setExcelFile(null);
      setRecipientEmail('');
    } catch (error) {
      console.error('Error:', error);
      setMessage(error.response?.data?.error || 'Error generating report. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Sprint Report Generator</h1>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="board-select">Board Name</label>
          <select 
            id="board-select"
            value={selectedBoard} 
            onChange={(e) => setSelectedBoard(e.target.value)}
            disabled={loadingBoards}
          >
            <option value="">Select Board</option>
            {boards.map(board => (
              <option key={board.id} value={board.id}>{board.name}</option>
            ))}
          </select>
          {loadingBoards && <div className="loading-text">Loading boards...</div>}
        </div>

        <div className="form-group">
          <label htmlFor="sprint-select">Sprint Name</label>
          <select 
            id="sprint-select"
            value={selectedSprint} 
            onChange={(e) => setSelectedSprint(e.target.value)}
            disabled={loadingSprints || !selectedBoard}
          >
            <option value="">Select Sprint</option>
            {sprints.map(sprint => (
              <option key={sprint.id} value={sprint.id}>{sprint.name}</option>
            ))}
          </select>
          {loadingSprints && <div className="loading-text">Loading sprints...</div>}
        </div>

        <div className="form-group">
          <label htmlFor="excel-file">Upload Sprint Capacity Sheet (Excel)</label>
          <input 
            id="excel-file"
            type="file" 
            accept=".xlsx,.xls" 
            onChange={handleFileChange}
            disabled={loading}
          />
          {excelFile && (
            <div className="file-name">
              Selected file: {excelFile.name}
            </div>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="email-input">Recipient Email</label>
          <input 
            id="email-input"
            type="email" 
            value={recipientEmail} 
            onChange={(e) => setRecipientEmail(e.target.value)}
            placeholder="Enter email address"
            disabled={loading}
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Generating Report...' : 'Generate Report'}
        </button>
      </form>

      {message && (
        <div className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
          {message}
        </div>
      )}
    </div>
  );
}

export default App; 